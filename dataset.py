"""
Dataset and data loading utilities for 5D YOLOv8 + GPS
"""
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import VOCDetection
from typing import Dict, List, Tuple

import config as cfg


class VOCExtended(Dataset):
    """
    Pascal-VOC dataset extended with synthetic depth, thermal, and GPS.
    """
    CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    def __init__(self, root=cfg.DATA_ROOT, split="train", year=cfg.DATASET_YEAR):
        self.voc = VOCDetection(root, year=year, image_set=split, download=True)
        self.to_tensor = T.ToTensor()
        self.resize = T.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE))
        self.cls_map = {name: idx for idx, name in enumerate(self.CLASSES)}

    def __len__(self) -> int:
        return len(self.voc)

    def _xywh_norm(self, bbox: Dict[str, str], width: int, height: int) -> Tuple[float, float, float, float]:
        """Convert VOC bbox format to normalized YOLO format."""
        xmin, ymin, xmax, ymax = map(float, (bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']))
        cx = (xmin + xmax) / 2 / width
        cy = (ymin + ymax) / 2 / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height
        return cx, cy, bw, bh  # all in 0-1 range

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict, torch.Tensor]:
        """Get dataset item with extended modalities."""
        img, ann = self.voc[idx]
        rgb = self.resize(self.to_tensor(img))                  # 3×H×W
        depth = torch.rand(1, cfg.IMG_SIZE, cfg.IMG_SIZE)       # Synthetic depth
        thermal = torch.rand(1, cfg.THERMAL_SIZE, cfg.THERMAL_SIZE) # Synthetic thermal
        rgbd = torch.cat([rgb, depth], 0)                       # 4×H×W

        # Extract image size for bbox normalization
        w0, h0 = map(int, (ann['annotation']['size']['width'],
                          ann['annotation']['size']['height']))

        # Process annotations
        cls, boxes = [], []
        for obj in ann['annotation']['object']:
            cls.append(self.cls_map[obj['name']])
            boxes.append(self._xywh_norm(obj['bndbox'], w0, h0))
        
        target = {
            "cls": torch.tensor(cls, dtype=torch.float32),      # (n,)
            "bboxes": torch.tensor(boxes, dtype=torch.float32)  # (n,4)
        }
        
        # Dummy GPS coordinates
        gps = torch.tensor([0.5, 0.5], dtype=torch.float32)
        
        return rgbd, thermal, target, gps


def collate_batch(batch: List[Tuple]) -> Tuple:
    """Custom collate function for dataloader."""
    rgbd, therm, target, gps = zip(*batch)
    return (
        torch.stack(rgbd),     # [B, 4, H, W]
        torch.stack(therm),    # [B, 1, Ht, Wt]
        list(target),          # List of B target dicts
        torch.stack(gps)       # [B, 2]
    )


def build_det_targets(target_list: List[Dict]) -> Dict[str, torch.Tensor]:
    """Convert per-image target dictionaries to batch format for v8DetectionLoss."""
    cls_list, boxes_list, batch_idx_list = [], [], []
    
    for i, target in enumerate(target_list):
        n = target['cls'].shape[0]
        cls_list.append(target['cls'].unsqueeze(1))             # (n, 1)
        boxes_list.append(target['bboxes'])                     # (n, 4)
        batch_idx_list.append(torch.full((n, 1), i, dtype=torch.float32))
    
    return {
        "cls": torch.vstack(cls_list).to(cfg.DEVICE),           # (N, 1)
        "bboxes": torch.vstack(boxes_list).to(cfg.DEVICE),      # (N, 4)
        "batch_idx": torch.vstack(batch_idx_list).to(cfg.DEVICE)  # (N, 1)
    }


def get_dataloaders() -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders."""
    # Create dataset
    dataset = VOCExtended(split="train")
    
    # Split into train/val
    train_size = int((1 - cfg.VAL_SPLIT) * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.SEED)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader