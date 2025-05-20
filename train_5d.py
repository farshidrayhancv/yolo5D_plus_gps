"""
train_5d.py

Demo: augment a pre-trained YOLOv8-n detector with depth, thermal and GPS
while keeping the core YOLO weights frozen.

* Dataset  : Pascal VOC (synthetic depth / thermal, dummy GPS)
* Backbone : YOLOv8-n (ultralytics)
* Heads    : - RGBD adapter  (4-ch → 3-ch)
             - Thermal fusion (adds thermal into mid features)
             - GPS regression (predicts 2-D coords)

Author: ChatGPT example, 2025-05-20
"""

# ───────────────────────── Imports ──────────────────────────
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCDetection
from tqdm import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ─────────────────── Global configuration ───────────────────
torch.manual_seed(42)             # Reproducibility
np.random.seed(42)
random.seed(42)

BATCH_SIZE    = 2
LR            = 1e-3
NUM_EPOCHS    = 5
IMG_SIZE      = 320
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ───────────────────────── Dataset ──────────────────────────
class VOCExtended(Dataset):
    """
    Pascal-VOC detection dataset *extended* with:
      • 1-channel synthetic depth map (same resolution as RGB)
      • 1-channel 96×96 synthetic thermal map
      • dummy GPS target (constant [45,45] for illustration)

    Bounding boxes are converted to YOLO centre-format and
    scaled to IMG_SIZE so the model sees consistent geometry.
    """
    def __init__(self, root="./data", year="2012",
                 image_set="train", download=True):
        self.voc = VOCDetection(root, year, image_set,
                                download=download, transform=None)

        print(f"VOC {year} {image_set} loaded ({len(self.voc)} images)")

        # Pascal classes + background
        self.classes = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
            'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
            'sofa', 'train', 'tvmonitor'
        ]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.to_tensor = transforms.ToTensor()
        self.resize    = transforms.Resize((IMG_SIZE, IMG_SIZE))

    def __len__(self): return len(self.voc)

    def __getitem__(self, idx):
        img, ann = self.voc[idx]

        # RGB tensor HWC→CHW, then resize
        rgb = self.resize(self.to_tensor(img))          # (3,H,W)

        # Synthetic modalities
        depth   = torch.rand(1, IMG_SIZE, IMG_SIZE)     # (1,H,W)
        thermal = torch.rand(1, 96, 96)                 # (1,96,96)

        rgbd = torch.cat([rgb, depth], dim=0)           # (4,H,W)

        # ── Convert VOC bboxes to centre-norm format ──
        w0 = float(ann['annotation']['size']['width'])
        h0 = float(ann['annotation']['size']['height'])
        boxes, labels = [], []

        for obj in ann['annotation']['object']:
            cls = self.class_to_idx.get(obj['name'], 0)
            bb  = obj['bndbox']
            xmin, ymin, xmax, ymax = map(float,
                                         (bb['xmin'], bb['ymin'],
                                          bb['xmax'], bb['ymax']))

            # Rescale to IMG_SIZE
            xmin, xmax = xmin / w0 * IMG_SIZE, xmax / w0 * IMG_SIZE
            ymin, ymax = ymin / h0 * IMG_SIZE, ymax / h0 * IMG_SIZE
            cx, cy = (xmin + xmax) / 2 / IMG_SIZE, (ymin + ymax) / 2 / IMG_SIZE
            bw, bh = (xmax - xmin) / IMG_SIZE, (ymax - ymin) / IMG_SIZE

            if bw > 0 and bh > 0:           # discard degenerate boxes
                boxes.append([cx, cy, bw, bh])
                labels.append(cls)

        if not boxes:                       # ensure ≥1 box (avoids sampler issues)
            boxes, labels = [[.5,.5,.1,.1]], [0]

        sample = {
            "boxes"      : torch.tensor(boxes,  dtype=torch.float32),
            "labels"     : torch.tensor(labels, dtype=torch.int64),
            "image_id"   : torch.tensor([idx]),
            "gps_coords" : torch.tensor([45.0, 45.0], dtype=torch.float32),
            "thermal_map": thermal                         # (1,96,96)
        }
        return rgbd, sample

# ──────────────────── Model building blocks ─────────────────
class RGBD2RGB(nn.Module):
    """1×1 conv turning 4-ch RGB-D into ordinary 3-ch RGB."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 3, 1, bias=False)

        # Initialise so RGB passes through untouched and
        # depth contributes only 10 % of its value.
        with torch.no_grad():
            w = torch.zeros(3,4,1,1)
            w[0,0] = w[1,1] = w[2,2] = 1.0   # identity for RGB
            w[:,3] = 0.1                     # small depth weight
            self.conv.weight.copy_(w)

    def forward(self, x): return self.conv(x)


class ThermalProcessor(nn.Module):
    """
    Embeds a 1-ch thermal map to match YOLO mid-feature channels
    then upsamples to the correct spatial resolution.
    """
    def __init__(self, out_ch=64):
        super().__init__()
        self.resize = nn.AdaptiveAvgPool2d((40, 40))   # 96×96 → 40×40
        self.conv   = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(16, out_ch, 1)
        )

    def forward(self, x, target_hw=None):
        x = self.resize(x)
        x = self.conv(x)
        if target_hw is not None:                      # match mid feature size
            x = nn.functional.interpolate(
                    x, size=target_hw, mode="bilinear",
                    align_corners=False)
        return x


class GPSHead(nn.Module):
    """
    Flexible 2-D regression head:
        • supports (B,C,H,W) or (B,L,C) input
        • simple GAP → FC → ReLU → FC
    """
    def __init__(self, in_ch: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.seq  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, 64),
            nn.ReLU(True),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # Handle different hook output layouts
        if isinstance(x, (list, tuple)):
            x = [t for t in x if isinstance(t, torch.Tensor)][-1]

        if x.dim() == 4:                 # (B,C,H,W)
            x = self.pool(x)
        elif x.dim() == 3:               # (B,L,C)
            x = x.mean(1, keepdim=True)  # (B,1,C)
        else:
            raise ValueError(f"GPSHead input ndim {x.dim()} not supported")
        return self.seq(x)

# ───────────────────────── YOLO-5D ─────────────────────────
class YOLO5D(nn.Module):
    """
    Wrapper around a frozen YOLOv8-n that:
      • Injects depth via RGBD2RGB
      • Fuses thermal into mid features
      • Adds a lightweight head to regress GPS
    """
    def __init__(self):
        super().__init__()
        self.rgbd_adapter   = RGBD2RGB()
        self.thermal_proc   = ThermalProcessor(64)
        self.gps_head       = GPSHead(256)     # placeholder (will re-init)

        # ----- Load pre-trained YOLOv8-n -----
        self.yolo = YOLO('yolov8n.pt')
        self.backbone = self.yolo.model
        self.backbone.eval()                  # we’ll keep it frozen

        # Feature capture
        self.feats, self.hooks = {}, []
        self._register_hooks()
        self._configure_aux_heads()           # infer channel sizes

    # ── register forward hooks to grab mid & final outputs
    def _register_hooks(self):
        def save(name):
            def _hook(_, __, out): self.feats[name] = out
            return _hook
        self.hooks.append(self.backbone.model[6]
                          .register_forward_hook(save("mid")))
        self.hooks.append(self.backbone.model[-1]
                          .register_forward_hook(save("final")))
        print("Registered feature extraction hooks")

    # ── run dummy input once so we know feature shapes
    def _configure_aux_heads(self):
        self.feats.clear()
        with torch.no_grad():
            dummy = torch.zeros(1, 4, IMG_SIZE, IMG_SIZE)
            self.backbone(self.rgbd_adapter(dummy))

        # Shape utilities
        def unwrap(x):
            if isinstance(x, (list, tuple)):
                x = [t for t in x if isinstance(t, torch.Tensor)][-1]
            return x

        mid = unwrap(self.feats["mid"])
        fin = unwrap(self.feats["final"])

        # Rebuild thermal proc to match mid channels/size
        _, c_mid, h_mid, w_mid = mid.shape
        self.thermal_proc = ThermalProcessor(c_mid)
        self.thermal_proc.resize = nn.AdaptiveAvgPool2d((h_mid, w_mid))

        # Re-init GPS head with correct channel count
        fin_ch = fin.shape[1] if fin.dim() == 4 else fin.shape[-1]
        self.gps_head = GPSHead(fin_ch)

        print("Auxiliary heads configured")

    # ─────────────────── forward ────────────────────
    def forward(self, x_rgbd, x_thermal=None):
        self.feats.clear()

        # 1. Adapt 4-ch → 3-ch and feed through YOLO
        rgb = self.rgbd_adapter(x_rgbd)
        detections = self.backbone(rgb)       # gradient flows to adapter only

        # 2. Fuse thermal map into mid features
        if x_thermal is not None and "mid" in self.feats:
            mid = self.feats["mid"]
            if isinstance(mid, (list, tuple)):
                mid = [t for t in mid if isinstance(t, torch.Tensor)][-1]
            h, w = mid.shape[2:]
            therm = self.thermal_proc(x_thermal, (h, w))
            self.feats["mid_fused"] = mid + 0.1 * therm

        # 3. GPS regression from final features
        gps = torch.tensor([[45.0, 45.0]] * x_rgbd.size(0),
                           device=x_rgbd.device)
        if "final" in self.feats:
            fin = self.feats["final"]
            gps = self.gps_head(fin)

        return {"detection": detections, "gps_coords": gps}

    # -- make sure we don’t accidentally train YOLO weights
    def train(self, mode=True):
        self.training = mode
        self.rgbd_adapter.train(mode)
        self.thermal_proc.train(mode)
        self.gps_head.train(mode)
        return self

    def eval(self): return self.train(False)

    def __del__(self):
        for h in self.hooks: h.remove()

# ───────────────────── Training utils ──────────────────────
def train_one_epoch(model, loader, opt, loss_fn):
    model.train()
    total, batches = 0., 0
    pbar = tqdm(loader, desc="Train")
    for rgbd, tgt in pbar:
        rgbd = torch.stack([x.to(DEVICE) for x in rgbd])
        thermal = torch.stack([t["thermal_map"].to(DEVICE) for t in tgt])
        gps_t   = torch.stack([t["gps_coords"].to(DEVICE) for t in tgt])

        out  = model(rgbd, thermal)
        loss = loss_fn(out["gps_coords"], gps_t)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.item(); batches += 1
        pbar.set_postfix({"gps_loss": total/batches})
        if batches >= 25: break              # shorten demo epoch
    return total / batches


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total_l, total_e, batches = 0., 0., 0
    for rgbd, tgt in loader:
        rgbd = torch.stack([x.to(DEVICE) for x in rgbd])
        thermal = torch.stack([t["thermal_map"].to(DEVICE) for t in tgt])
        gps_t   = torch.stack([t["gps_coords"].to(DEVICE) for t in tgt])

        out  = model(rgbd, thermal)
        total_l += loss_fn(out["gps_coords"], gps_t).item()
        total_e += torch.abs(out["gps_coords"] - gps_t).mean().item()
        batches += 1
        if batches >= 10: break
    return total_l/batches, total_e/batches


def train_model(model, train_loader, val_loader, epochs=NUM_EPOCHS):
    loss_fn  = nn.MSELoss()
    opt = optim.Adam(
        list(model.rgbd_adapter.parameters()) +
        list(model.thermal_proc.parameters()) +
        list(model.gps_head.parameters()),
        lr=LR
    )

    best = float('inf')
    for ep in range(epochs):
        tr = train_one_epoch(model, train_loader, opt, loss_fn)
        vl, ve = evaluate(model, val_loader, loss_fn)
        print(f"Epoch {ep+1}/{epochs} | Train {tr:.4f} "
              f"| ValLoss {vl:.4f} | ValErr {ve:.4f}")

        if vl < best:
            best = vl
            torch.save(model.state_dict(), "best_5d_thermal_gps_model.pt")
            print("  ↳ saved new best")

    torch.save(model.state_dict(), "final_5d_thermal_gps_model.pt")
    return best

# ───────────────────── Data helpers ────────────────────────
def make_loaders(dataset):
    n = len(dataset); n_tr = int(.8*n)
    tr_ds, va_ds = torch.utils.data.random_split(dataset, [n_tr, n-n_tr])

    collate = lambda b: ([x[0] for x in b], [x[1] for x in b])
    tr = DataLoader(tr_ds, BATCH_SIZE, True,  collate_fn=collate)
    va = DataLoader(va_ds, BATCH_SIZE, False, collate_fn=collate)
    return tr, va

# ───────────────────── Visualisation ───────────────────────
@torch.no_grad()
def visualise(model, dataset, k=5):
    """Save RGB/depth/thermal triptychs with predicted GPS."""
    os.makedirs("results", exist_ok=True)
    model.eval()
    for i in range(min(k, len(dataset))):
        rgbd, tgt = dataset[i]
        rgbd_ = rgbd.unsqueeze(0).to(DEVICE)
        therm_ = tgt["thermal_map"].unsqueeze(0).to(DEVICE)
        out = model(rgbd_, therm_)["gps_coords"][0].cpu().numpy()
        true = tgt["gps_coords"].numpy()

        rgb = rgbd[:3].permute(1,2,0).numpy()
        rgb = (rgb - rgb.min()) / (rgb.ptp()+1e-8)
        depth = rgbd[3].numpy()

        fig, ax = plt.subplots(1,3, figsize=(18,6))
        # RGB + boxes
        ax[0].imshow(rgb); ax[0].set_title("RGB")
        for box, lab in zip(tgt["boxes"], tgt["labels"]):
            cx,cy,w,h = box
            x1,y1 = (cx-w/2)*IMG_SIZE, (cy-h/2)*IMG_SIZE
            ax[0].add_patch(
                plt.Rectangle((x1,y1), w*IMG_SIZE, h*IMG_SIZE,
                              fill=False, edgecolor='g', linewidth=2)
            )
        # Depth
        ax[1].imshow((depth-depth.min())/(depth.ptp()+1e-8), cmap='viridis')
        ax[1].set_title("Depth")
        # Thermal
        therm = therm_[0,0].cpu().numpy()
        ax[2].imshow((therm-therm.min())/(therm.ptp()+1e-8), cmap='inferno')
        ax[2].set_title("Thermal")

        fig.suptitle(f"True GPS {true} | Pred {out.round(2)}")
        plt.tight_layout(); fig.savefig(f"results/sample_{i+1}.png"); plt.close()

# ───────────────────────── Main ────────────────────────────
def main():
    print("Starting 5-D object detection demo…")
    ds = VOCExtended(download=True)
    tr_loader, va_loader = make_loaders(ds)

    print("Building model …")
    model = YOLO5D().to(DEVICE)

    print("Training …")
    train_model(model, tr_loader, va_loader, NUM_EPOCHS)

    # quick sanity check on first sample
    rgbd, tgt = ds[0]
    pred = model(rgbd.unsqueeze(0).to(DEVICE),
                 tgt["thermal_map"].unsqueeze(0).to(DEVICE))["gps_coords"][0]
    print(f"[Quick test] true {tgt['gps_coords'].numpy()} "
          f"pred {pred.cpu().numpy()}")

    visualise(model, ds, 5)
    print("Done. Results in ./results")

if __name__ == "__main__":
    main()
