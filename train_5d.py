"""
yolo5d_train.py ───────────────────────────────────────────────────────────────
End-to-end 5-D YOLOv8 training demo

• Input  : 4-channel RGB-D tensor  +  1-channel thermal map
• Fusion : thermal is injected into YOLO's mid-feature via a forward hook
• Output : standard YOLO detections  +  global 2-D GPS vector
• Loss   : Ultralytics v8 detection loss  +  MSE GPS loss
• All weights (backbone + new layers) are trainable
───────────────────────────────────────────────────────────────────────────────
Tested with:
    pip install ultralytics==8.3.140 torch>=2.1 torchvision>=0.16
"""
# ────────────────────────── IMPORTS ──────────────────────────
from pathlib import Path
import random, numpy as np, torch
import torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data   import Dataset, DataLoader, random_split
from torchvision.datasets import VOCDetection
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss          # detection loss
from types import SimpleNamespace   #  add to the imports at top



# ───────────────────── CONFIG / SEEDS ───────────────────────
SEED = 42
IMG_SIZE   = 320            # square resize for VOC
BATCH_SIZE = 8
EPOCHS     = 30
LR_BACKBONE = 1e-4          # slower LR for pre-trained layers
LR_NEW      = 1e-3          # higher LR for new branches
LAMBDA_GPS  = 1.0           # weight of GPS loss
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ───────────────────── DATASET ──────────────────────────────
class VOCExtended(Dataset):
    """Pascal-VOC with synthetic depth + thermal + dummy GPS (lat,lon)"""
    CLASSES = [
        'aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair',
        'cow','diningtable','dog','horse','motorbike','person','pottedplant',
        'sheep','sofa','train','tvmonitor'
    ]
    def __init__(self, root="data", split="train", year="2012"):
        self.voc = VOCDetection(root, year=year, image_set=split, download=True)
        self.to_tensor = T.ToTensor()
        self.resize    = T.Resize((IMG_SIZE, IMG_SIZE))
        self.cls_map   = {n:i for i,n in enumerate(self.CLASSES)}

    def __len__(self): return len(self.voc)

    def _xywh_norm(self, bb, w0, h0):
        xmin,ymin,xmax,ymax = map(float,(bb['xmin'],bb['ymin'],bb['xmax'],bb['ymax']))
        cx = (xmin + xmax) / 2 / w0
        cy = (ymin + ymax) / 2 / h0
        bw = (xmax - xmin) / w0
        bh = (ymax - ymin) / h0
        return cx, cy, bw, bh   # all in 0-1

    def __getitem__(self, idx):
        img, ann = self.voc[idx]
        rgb  = self.resize(self.to_tensor(img))          # 3×H×W
        depth   = torch.rand(1, IMG_SIZE, IMG_SIZE)      # fake depth
        thermal = torch.rand(1, 96, 96)                  # fake thermal
        rgbd = torch.cat([rgb, depth], 0)                # 4×H×W

        w0, h0 = map(int, (ann['annotation']['size']['width'],
                           ann['annotation']['size']['height']))
        cls, boxes = [], []
        for obj in ann['annotation']['object']:
            cls.append(self.cls_map[obj['name']])
            boxes.append(self._xywh_norm(obj['bndbox'], w0, h0))
        target = {
            "cls":    torch.tensor(cls,   dtype=torch.float32),   # (n,)
            "bboxes": torch.tensor(boxes, dtype=torch.float32)    # (n,4)
        }
        gps = torch.tensor([0.5, 0.5], dtype=torch.float32)       # dummy
        return rgbd, thermal, target, gps

# ────────────────── MODEL BUILDING BLOCKS ──────────────────
class RGBD2RGB(nn.Module):
    """1×1 conv: 4-ch → 3-ch (learnable)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4,3,1,bias=False)
        with torch.no_grad():
            w = torch.zeros(3,4,1,1)
            w[0,0]=w[1,1]=w[2,2]=1.; w[:,3]=0.1
            self.conv.weight.copy_(w)
    def forward(self,x): return self.conv(x)

class ThermalProcessor(nn.Module):
    """Embed + upsample thermal map to match mid-feature size."""
    def __init__(self, out_ch:int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,1,1), nn.ReLU(True),
            nn.Conv2d(16,out_ch,1))
    def forward(self,t,h,w):
        t = F.interpolate(t,(h,w),mode="bilinear",align_corners=False)
        return self.net(t)

class GPSHead(nn.Module):
    """Dynamic MLP: infer input dim at first forward."""
    def __init__(self, hidden=128):
        super().__init__()
        self.fc1 = None
        self.hidden = hidden
        self.relu  = nn.ReLU(inplace=True)
        self.fc2   = nn.Linear(hidden, 2)
        self.sig   = nn.Sigmoid()
    def forward(self, x):
        if x.dim()==4: x = x.mean((2,3))
        elif x.dim()==3: x = x.mean(1)
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(-1), self.hidden).to(x.device)
            # Move fc2 to match fc1's device
            self.fc2 = self.fc2.to(x.device)
        x = self.relu(self.fc1(x))
        return self.sig(self.fc2(x))

# ───────────────────── YOLO-5D MODEL ───────────────────────
class YOLO5D(nn.Module):
    """YOLOv8-n with RGB-D adapter, thermal fusion, GPS regression."""
    def __init__(self, weights="yolov8n.pt"):
        super().__init__()
        # Initialize on CPU first
        yolo = YOLO(weights)
        self.backbone = yolo.model
        self.det_loss = v8DetectionLoss(self.backbone)

        # Supply the minimal hyper-parameter namespace it expects
        default_hyp = dict(box=7.5, cls=0.5, dfl=1.5,
                           obj=1.0, pose=12.0, landmark=1.0, kpt=1.0)
        self.det_loss.hyp = SimpleNamespace(**default_hyp)
        
        # Initialize adapters
        self.adapt = RGBD2RGB()
        
        # Create dummy input
        dummy = torch.zeros(1, 4, IMG_SIZE, IMG_SIZE)
        
        # Probe mid channels
        with torch.no_grad():
            mid_dict = {}
            h = self.backbone.model[6].register_forward_hook(
                lambda m, i, o: mid_dict.setdefault("feat", o))
            _ = self.backbone(self.adapt(dummy))
            h.remove()
        
        mid_ch = mid_dict["feat"].shape[1]
        
        # Initialize thermal processor and GPS head
        self.t_fuse = ThermalProcessor(mid_ch)
        self.gps_head = GPSHead()
        
        # Forward-pre-hook: add thermal into layer-6 output
        self._cur_therm = None
        def fuse_hook(mod, inp):
            x = inp[0]
            if self._cur_therm is None: return
            h, w = x.shape[-2:]
            return (x + 0.1 * self.t_fuse(self._cur_therm, h, w),)
        
        self.backbone.model[6].register_forward_pre_hook(fuse_hook)
        

    # standard train / eval toggles
    def train(self, mode=True):
        super().train(mode)
        self.backbone.train(mode)
        return self

    # full forward: returns raw detection tensors + gps
    def forward(self, rgbd, thermal):
        # Store thermal for hook
        self._cur_therm = thermal
        
        # Forward pass through backbone
        preds = self.backbone(self.adapt(rgbd))
        
        # Clear thermal reference
        self._cur_therm = None
        
        # Apply GPS head to the first prediction tensor
        first_pred = preds[0] if isinstance(preds, list) else preds
        gps = self.gps_head(first_pred.flatten(1, -2))
        
        return preds, gps

    def eval(self):
        """
        Override .eval() so the backbone remains in training mode.
        This guarantees raw predictions (no post-processing), while
        freezing batch-norm & dropout in our adapters / GPS head.
        """
        super().eval()                 # sets all modules to eval
        self.backbone.train(True)      # but keep YOLO in train-mode
        return self

    def to(self, *args, **kwargs):
        """
        Override nn.Module.to():
        1. move the whole model as usual (super().to)
        2. move *all* tensors/buffers inside self.det_loss to the same device
        3. update self.det_loss.device so its internal code builds tensors on-device
        """
        super().to(*args, **kwargs)
        dev = next(self.parameters()).device

        dl = self.det_loss                       # convenience
        dl.device = dev                          # <-- critical

        # Walk through the attributes once and move what’s a Tensor / list[Tensor]
        for name, val in dl.__dict__.items():
            if torch.is_tensor(val):
                setattr(dl, name, val.to(dev, non_blocking=True))
            elif isinstance(val, (list, tuple)) and val and torch.is_tensor(val[0]):
                setattr(dl, name, [v.to(dev, non_blocking=True) for v in val])

        return self
    
    @torch.no_grad()
    def predict(self, rgbd, thermal, conf=0.25, iou=0.7):
        """
        Runs with Ultralytics post-processing for deployment.
        Returns: results (ultralytics.engine.results.Results)  and  gps tensor
        """
        self.eval()                          # sets eval, but keep backbone train=True (see below)
        # forward once to get raw + GPS
        preds, gps = self(rgbd, thermal)     # raw tensors, gps
        # post-process detections the same way Ultralytics does internally
        results = self.backbone.model[-1].new_results(preds, rgbd, 
                                                     proto=None,
                                                     imgsz=IMG_SIZE,
                                                     conf=conf, iou=iou)
        return results, gps


# ──────────────── DATA LOADERS & COLLATE ───────────────────
def build_loaders():
    ds = VOCExtended(split="train")
    n_tr = int(0.9*len(ds))
    tr, va = random_split(ds,[n_tr,len(ds)-n_tr],
                          generator=torch.Generator().manual_seed(SEED))
    def collate(batch):
        rgbd, therm, target, gps = zip(*batch)
        return (torch.stack(rgbd), torch.stack(therm),
                list(target), torch.stack(gps))
    return (DataLoader(tr,BATCH_SIZE,True,collate_fn=collate,num_workers=4),
            DataLoader(va,BATCH_SIZE,False,collate_fn=collate,num_workers=4))

# ───────────────── TARGET BUILDER ──────────────────────────
def build_det_targets(tgt_list):
    """merge per-image dicts into single batch dict for v8DetectionLoss."""
    cls, boxes, bidx = [], [], []
    for i,t in enumerate(tgt_list):
        n = t['cls'].shape[0]
        cls.append(t['cls'].unsqueeze(1))          # (n,1)
        boxes.append(t['bboxes'])                  # (n,4) in 0-1
        bidx.append(torch.full((n,1), i, dtype=torch.float32))
    return {
        "cls":   torch.vstack(cls).to(DEVICE),     # (N,1)
        "bboxes":torch.vstack(boxes).to(DEVICE),   # (N,4)
        "batch_idx": torch.vstack(bidx).to(DEVICE) # (N,1)
    }

# ───────────────── TRAIN & VAL LOOPS ───────────────────────
def make_optim(model):
    back,new = [],[]
    for n,p in model.named_parameters():
        (new if any(k in n for k in ['adapt','t_fuse','gps_head'])
         else back).append(p)
    return optim.Adam([
        {"params":back, "lr":LR_BACKBONE},
        {"params":new , "lr":LR_NEW     }])

def train_epoch(model, loader, optimiser):
    model.train()
    pbar = tqdm(loader, desc="Train")
    for rgbd, therm, tgt, gps in pbar:
        # Move data to the same device as the model
        rgbd, therm, gps = rgbd.to(DEVICE), therm.to(DEVICE), gps.to(DEVICE)
        
        # Forward pass
        preds, gps_pred = model(rgbd, therm)
        
        # Build detection targets
        det_tgt = build_det_targets(tgt)
        
        # Ensure det_loss has its internal tensors on the correct device
        if hasattr(model.det_loss, 'proj') and model.det_loss.proj.device != DEVICE:
            model.det_loss.proj = model.det_loss.proj.to(DEVICE)
            
        # Calculate losses
        det_loss, _ = model.det_loss(preds, det_tgt)
        gps_loss = F.mse_loss(gps_pred, gps)
        loss = det_loss + LAMBDA_GPS * gps_loss
        
        # Backprop
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        pbar.set_postfix(det=det_loss.item(), gps=gps_loss.item())

@torch.no_grad()
def val_epoch(model, loader):
    model.eval()
    det_tot = gps_tot = n = 0
    for rgbd, therm, tgt, gps in loader:
        # Move data to the same device as the model
        rgbd, therm, gps = rgbd.to(DEVICE), therm.to(DEVICE), gps.to(DEVICE)
        
        # Forward pass
        preds, gps_pred = model(rgbd, therm)
        
        # Calculate losses
        det_tgt = build_det_targets(tgt)
        det_loss, _ = model.det_loss(preds, det_tgt)
        gps_loss = F.mse_loss(gps_pred, gps)
        
        # Accumulate batch statistics
        b = rgbd.size(0)
        det_tot += det_loss.item() * b
        gps_tot += gps_loss.item() * b
        n += b
        
    return det_tot/n, gps_tot/n

# ───────────────────────── MAIN ────────────────────────────
# ───────────────────────── MAIN ────────────────────────────
def main():
    # Get data loaders
    tr_loader, va_loader = build_loaders()
    
    # Print device information
    print(f"Using device: {DEVICE}")
    
    # Create model first (initializes on CPU)
    model = YOLO5D()
    
    # Move ENTIRE model to the target device at once
    model = model.to(DEVICE)
    
    # Also explicitly move the det_loss's internal projector tensor
    if hasattr(model.det_loss, 'proj'):
        model.det_loss.proj = model.det_loss.proj.to(DEVICE)
    if hasattr(model.det_loss, 'stride'):
        model.det_loss.stride = model.det_loss.stride.to(DEVICE)
    if hasattr(model.det_loss, 'anchors'):
        model.det_loss.anchors = model.det_loss.anchors.to(DEVICE)
    
    # Initialize optimizer after model is on the correct device
    optimiser = make_optim(model)
    Path("ckpts").mkdir(exist_ok=True)

    best_val = 1e9
    for epoch in range(1, EPOCHS+1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        train_epoch(model, tr_loader, optimiser)
        det_v, gps_v = val_epoch(model, va_loader)
        print(f"Val  det_loss {det_v:.3f} | gps_loss {gps_v:.2f}")
        if det_v+gps_v < best_val:
            best_val = det_v+gps_v
            torch.save(model.state_dict(), "ckpts/yolo5d_best.pt")
            print("  ↳ saved new best checkpoint")

    torch.save(model.state_dict(), "ckpts/yolo5d_final.pt")
    print("Training finished")

if __name__ == "__main__":
    main()
