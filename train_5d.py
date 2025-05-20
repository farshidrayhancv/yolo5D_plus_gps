"""
train_5d.py - YOLOv8 Multi-Modal Object Detection with GPS Regression
===========================================================================

A PyTorch implementation that extends Ultralytics YOLOv8 for:
    • Multi-modal input: RGB + Depth + Thermal (5 channels total)
    • Thermal fusion: integrated at mid-backbone layer via hooks
    • Dual-head output: standard object detection + GPS coordinates
    • Joint training: detection loss + GPS regression loss

Architecture:
    • RGB-D (4ch) → Adapter (1x1 conv) → YOLOv8 Backbone
    • Thermal (1ch) → ThermalCNN → injected into backbone via hooks
    • YOLOv8 Features → [Detection Head, GPS Head]

Usage:
    $ python train_5d.py  # trains on VOC with synthetic depth, thermal, GPS

Requirements:
    • Python 3.8+
    • PyTorch 2.1+
    • Ultralytics 8.3.x
"""
import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import VOCDetection
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss

# ======================================================================
# Configuration
# ======================================================================

class Config:
    """Configuration parameters for training and model architecture."""
    # Model parameters
    IMG_SIZE = 320            # Square input size
    THERMAL_SIZE = 96         # Thermal input size
    THERMAL_WEIGHT = 0.1      # Weight for thermal feature fusion
    
    # Training parameters
    SEED = 42                 # Random seed for reproducibility
    BATCH_SIZE = 8            # Training batch size
    EPOCHS = 30               # Number of training epochs
    LR_BACKBONE = 1e-4        # Learning rate for pre-trained backbone
    LR_NEW = 1e-3             # Learning rate for new layers
    LAMBDA_GPS = 1.0          # GPS loss weight
    
    # Data parameters
    DATA_ROOT = "data"        # Dataset root directory
    
    # Inference parameters
    CONF_THRESH = 0.25        # Confidence threshold for NMS
    IOU_THRESH = 0.7          # IoU threshold for NMS

    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Set random seeds for reproducibility
def set_seeds(seed: int = Config.SEED) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# ======================================================================
# Dataset
# ======================================================================

class VOCExtended(Dataset):
    """
    Pascal-VOC dataset extended with synthetic depth, thermal, and GPS.
    
    This dataset wrapper augments VOCDetection with:
        - Synthetic depth channel (1×H×W)
        - Synthetic thermal data (1×96×96)
        - Dummy GPS coordinates [0.5, 0.5]
    
    Args:
        root: Root directory for dataset storage
        split: Dataset split ('train' or 'val')
        year: VOC dataset year
    """
    CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    def __init__(self, root: str = Config.DATA_ROOT, split: str = "train", year: str = "2012"):
        self.voc = VOCDetection(root, year=year, image_set=split, download=True)
        self.to_tensor = T.ToTensor()
        self.resize = T.Resize((Config.IMG_SIZE, Config.IMG_SIZE))
        self.cls_map = {name: idx for idx, name in enumerate(self.CLASSES)}

    def __len__(self) -> int:
        return len(self.voc)

    def _xywh_norm(self, bbox: Dict[str, str], width: int, height: int) -> Tuple[float, float, float, float]:
        """
        Convert VOC bbox format (xmin, ymin, xmax, ymax) to normalized YOLO format (cx, cy, w, h).
        
        Args:
            bbox: Dictionary with 'xmin', 'ymin', 'xmax', 'ymax' keys
            width: Original image width
            height: Original image height
            
        Returns:
            Tuple of normalized (center_x, center_y, width, height)
        """
        xmin, ymin, xmax, ymax = map(float, (bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']))
        cx = (xmin + xmax) / 2 / width
        cy = (ymin + ymax) / 2 / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height
        return cx, cy, bw, bh  # all in 0-1 range

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict, torch.Tensor]:
        """
        Get dataset item with extended modalities.
        
        Args:
            idx: Dataset index
        
        Returns:
            Tuple containing:
                - RGB-D tensor (4×H×W)
                - Thermal tensor (1×96×96)
                - Target dictionary with 'cls' and 'bboxes' keys
                - GPS tensor [lat, lon]
        """
        img, ann = self.voc[idx]
        rgb = self.resize(self.to_tensor(img))                  # 3×H×W
        depth = torch.rand(1, Config.IMG_SIZE, Config.IMG_SIZE) # Synthetic depth
        thermal = torch.rand(1, Config.THERMAL_SIZE, Config.THERMAL_SIZE) # Synthetic thermal
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


# ======================================================================
# Model Building Blocks
# ======================================================================

class RGBD2RGB(nn.Module):
    """
    Learnable adapter to convert 4-channel RGB-D to 3-channel input.
    
    Uses a 1×1 convolution to project 4 channels to 3 channels.
    Initialized with identity weights for RGB and small weight for depth.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 3, kernel_size=1, bias=False)
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights to identity for RGB channels and small value for depth."""
        with torch.no_grad():
            w = torch.zeros(3, 4, 1, 1)
            # Identity mapping for RGB channels
            w[0, 0] = w[1, 1] = w[2, 2] = 1.0
            # Small initial weight for depth channel
            w[:, 3] = 0.1
            self.conv.weight.copy_(w)
            
    def forward(self, x):
        return self.conv(x)


class ThermalProcessor(nn.Module):
    """
    Process and upsample thermal input to match feature dimensions.
    
    Args:
        out_channels: Number of output channels to match backbone feature size
    """
    def __init__(self, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1)
        )
        
    def forward(self, thermal: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Process and upsample thermal input.
        
        Args:
            thermal: Thermal input tensor
            height: Target feature height
            width: Target feature width
            
        Returns:
            Processed thermal features with shape [B, out_channels, H, W]
        """
        # Upsample to target dimensions
        thermal_resized = F.interpolate(
            thermal, size=(height, width), 
            mode="bilinear", align_corners=False
        )
        # Process through convolution network
        return self.net(thermal_resized)


class GPSHead(nn.Module):
    """
    GPS regression head with dynamic input dimension.
    
    Infers input dimension on first forward pass and creates appropriate
    fully connected layers. Output is normalized to [0,1] range.
    
    Args:
        hidden_dim: Dimension of hidden layer
    """
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = None  # Will be initialized on first forward pass
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, 2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GPS head.
        
        Args:
            x: Input tensor, can be 4D (B,C,H,W) or 3D (B,N,C)
            
        Returns:
            GPS coordinates tensor of shape [B, 2]
        """
        # Global average pooling if needed
        if x.dim() == 4:  # (B,C,H,W)
            x = x.mean((2, 3))
        elif x.dim() == 3:  # (B,N,C)
            x = x.mean(1)
            
        # Lazily initialize FC1 with correct input dimension
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(-1), self.hidden_dim).to(x.device)
            # Move fc2 to match fc1's device
            self.fc2 = self.fc2.to(x.device)
            
        # Forward pass
        x = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))


# ======================================================================
# YOLO5D Model
# ======================================================================

class YOLO5D(nn.Module):
    """
    YOLOv8 model extended for RGB-D-Thermal input and GPS regression.
    
    Features:
    - Takes 4-channel RGB-D input and 1-channel thermal input
    - Adapts RGB-D to RGB format for YOLOv8 backbone
    - Injects thermal features into mid-level backbone features
    - Adds GPS regression head alongside detection head
    
    Args:
        weights: Path or name of YOLOv8 weights for initialization
    """
    def __init__(self, weights: str = "yolov8n.pt"):
        super().__init__()
        # Load base YOLO model (on CPU first)
        yolo = YOLO(weights)
        self.backbone = yolo.model
        self.det_loss = v8DetectionLoss(self.backbone)
        
        # Configure detection loss hyperparameters
        default_hyp = {
            "box": 7.5, "cls": 0.5, "dfl": 1.5,
            "obj": 1.0, "pose": 12.0, "landmark": 1.0, "kpt": 1.0
        }
        self.det_loss.hyp = SimpleNamespace(**default_hyp)
        
        # Initialize RGB-D adapter
        self.adapt = RGBD2RGB()
        
        # Determine feature dimensions at layer 6 (mid-backbone)
        mid_channels = self._probe_mid_features()
        
        # Initialize thermal processor and GPS head
        self.t_fuse = ThermalProcessor(mid_channels)
        self.gps_head = GPSHead()
        
        # Set up thermal fusion hook
        self._cur_therm = None  # Current thermal tensor reference
        self._register_fusion_hook()
    
    def _probe_mid_features(self) -> int:
        """
        Probe mid-level backbone features to determine channel count.
        
        Returns:
            Number of channels in mid-level feature map
        """
        # Create dummy input
        dummy = torch.zeros(1, 4, Config.IMG_SIZE, Config.IMG_SIZE)
        
        # Capture mid-level features
        mid_features = {}
        
        def capture_hook(module, inp, output):
            mid_features["feat"] = output
            return output
        
        # Register temporary hook
        handle = self.backbone.model[6].register_forward_hook(capture_hook)
        
        # Perform dummy forward pass
        with torch.no_grad():
            _ = self.backbone(self.adapt(dummy))
        
        # Remove hook
        handle.remove()
        
        # Return channel count
        return mid_features["feat"].shape[1]
    
    def _register_fusion_hook(self):
        """Register hook to fuse thermal features into backbone."""
        def fusion_pre_hook(module, inputs):
            """Pre-forward hook for thermal fusion."""
            x = inputs[0]
            
            # Skip fusion if no thermal tensor is available
            if self._cur_therm is None:
                return
            
            # Get spatial dimensions
            h, w = x.shape[-2:]
            
            # Process thermal and add to feature map
            thermal_features = self.t_fuse(self._cur_therm, h, w)
            return (x + Config.THERMAL_WEIGHT * thermal_features,)
        
        # Register the hook at layer 6
        self.backbone.model[6].register_forward_pre_hook(fusion_pre_hook)
    
    def forward(self, 
                rgbd: torch.Tensor, 
                thermal: torch.Tensor) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]:
        """
        Forward pass through complete model.
        
        Args:
            rgbd: RGB-D input tensor [B, 4, H, W]
            thermal: Thermal input tensor [B, 1, Ht, Wt]
            
        Returns:
            Tuple containing:
                - Detection predictions (tensor or list of tensors)
                - GPS predictions [B, 2]
        """
        # Store thermal reference for hook
        self._cur_therm = thermal
        
        # Forward through adapter and backbone
        adapted = self.adapt(rgbd)
        preds = self.backbone(adapted)
        
        # Clear thermal reference
        self._cur_therm = None
        
        # Extract feature map for GPS head
        first_pred = preds[0] if isinstance(preds, list) else preds
        gps = self.gps_head(first_pred)
        
        return preds, gps
    
    def eval(self):
        """
        Override eval() to keep backbone in training mode.
        
        This ensures raw predictions (no NMS) while freezing batch-norm
        and dropout in adapters and GPS head.
        """
        super().eval()                 # Sets all modules to eval
        self.backbone.train(True)      # Keep YOLO in train-mode
        return self
    
    def to(self, *args, **kwargs):
        """
        Override to() to handle detection loss internals.
        
        Moves the entire model and ensures all detection loss tensors
        are on the same device.
        """
        # Move model
        device_obj = super().to(*args, **kwargs)
        
        # Get device from a parameter
        device = next(self.parameters()).device
        
        # Update detection loss device
        self.det_loss.device = device
        
        # Move tensors in detection loss
        for name, val in self.det_loss.__dict__.items():
            if torch.is_tensor(val):
                setattr(self.det_loss, name, val.to(device, non_blocking=True))
            elif isinstance(val, (list, tuple)) and val and torch.is_tensor(val[0]):
                setattr(self.det_loss, name, [v.to(device, non_blocking=True) for v in val])
        
        return device_obj
    
    @torch.no_grad()
    def predict(self, 
                rgbd: torch.Tensor, 
                thermal: torch.Tensor, 
                conf: float = Config.CONF_THRESH, 
                iou: float = Config.IOU_THRESH):
        """
        Run inference with post-processing for deployment.
        
        Args:
            rgbd: RGB-D input tensor [B, 4, H, W]
            thermal: Thermal input tensor [B, 1, Ht, Wt]
            conf: Confidence threshold for NMS
            iou: IoU threshold for NMS
            
        Returns:
            Tuple containing:
                - Processed detection results (ultralytics.engine.results.Results)
                - GPS predictions [B, 2]
        """
        # Set to eval mode
        self.eval()
        
        # Forward pass
        preds, gps = self(rgbd, thermal)
        
        # Post-process detections
        results = self.backbone.model[-1].new_results(
            preds, rgbd,
            proto=None,
            imgsz=Config.IMG_SIZE,
            conf=conf, iou=iou
        )
        
        return results, gps


# ======================================================================
# Data Loading Functions
# ======================================================================

def collate_batch(batch: List[Tuple]) -> Tuple:
    """
    Custom collate function for dataloader.
    
    Args:
        batch: List of dataset items
        
    Returns:
        Tuple containing batched tensors and list of targets
    """
    rgbd, therm, target, gps = zip(*batch)
    return (
        torch.stack(rgbd),     # [B, 4, H, W]
        torch.stack(therm),    # [B, 1, Ht, Wt]
        list(target),          # List of B target dicts
        torch.stack(gps)       # [B, 2]
    )


def build_loaders() -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create dataset
    dataset = VOCExtended(split="train")
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(Config.SEED)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader


def build_det_targets(target_list: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Convert per-image target dictionaries to batch format for v8DetectionLoss.
    
    Args:
        target_list: List of per-image target dictionaries
        
    Returns:
        Dictionary with batched tensors for detection loss
    """
    cls_list, boxes_list, batch_idx_list = [], [], []
    
    for i, target in enumerate(target_list):
        n = target['cls'].shape[0]
        cls_list.append(target['cls'].unsqueeze(1))             # (n, 1)
        boxes_list.append(target['bboxes'])                     # (n, 4)
        batch_idx_list.append(torch.full((n, 1), i, dtype=torch.float32))
    
    return {
        "cls": torch.vstack(cls_list).to(Config.DEVICE),        # (N, 1)
        "bboxes": torch.vstack(boxes_list).to(Config.DEVICE),   # (N, 4)
        "batch_idx": torch.vstack(batch_idx_list).to(Config.DEVICE)  # (N, 1)
    }


# ======================================================================
# Training Functions
# ======================================================================

def create_optimizer(model: YOLO5D) -> optim.Optimizer:
    """
    Create optimizer with different learning rates for different parts.
    
    Args:
        model: YOLO5D model
        
    Returns:
        Adam optimizer with parameter groups
    """
    backbone_params, new_params = [], []
    
    # Separate parameters into groups
    for name, param in model.named_parameters():
        if any(k in name for k in ['adapt', 't_fuse', 'gps_head']):
            new_params.append(param)
        else:
            backbone_params.append(param)
    
    # Create optimizer with different learning rates
    return optim.Adam([
        {"params": backbone_params, "lr": Config.LR_BACKBONE},
        {"params": new_params, "lr": Config.LR_NEW}
    ])


def train_epoch(model: YOLO5D, loader: DataLoader, optimizer: optim.Optimizer) -> None:
    """
    Train model for one epoch.
    
    Args:
        model: YOLO5D model
        loader: Training data loader
        optimizer: Optimizer
    """
    model.train()
    progress_bar = tqdm(loader, desc="Train")
    
    for rgbd, thermal, targets, gps in progress_bar:
        # Move data to device
        rgbd = rgbd.to(Config.DEVICE, non_blocking=True)
        thermal = thermal.to(Config.DEVICE, non_blocking=True)
        gps = gps.to(Config.DEVICE, non_blocking=True)
        
        # Forward pass
        preds, gps_pred = model(rgbd, thermal)
        
        # Build detection targets
        det_targets = build_det_targets(targets)
        
        # Ensure detection loss tensors are on the correct device
        if hasattr(model.det_loss, 'proj') and model.det_loss.proj.device != Config.DEVICE:
            model.det_loss.proj = model.det_loss.proj.to(Config.DEVICE)
        
        # Calculate losses
        det_loss, _ = model.det_loss(preds, det_targets)
        gps_loss = F.mse_loss(gps_pred, gps)
        total_loss = det_loss + Config.LAMBDA_GPS * gps_loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Update progress bar
        progress_bar.set_postfix(
            det=f"{det_loss.item():.4f}", 
            gps=f"{gps_loss.item():.6f}"
        )


@torch.no_grad()
def validate(model: YOLO5D, loader: DataLoader) -> Tuple[float, float]:
    """
    Validate model on validation dataset.
    
    Args:
        model: YOLO5D model
        loader: Validation data loader
        
    Returns:
        Tuple of (detection_loss, gps_loss)
    """
    model.eval()
    det_total = gps_total = sample_count = 0
    
    for rgbd, thermal, targets, gps in loader:
        # Move data to device
        rgbd = rgbd.to(Config.DEVICE, non_blocking=True)
        thermal = thermal.to(Config.DEVICE, non_blocking=True)
        gps = gps.to(Config.DEVICE, non_blocking=True)
        
        # Forward pass
        preds, gps_pred = model(rgbd, thermal)
        
        # Calculate losses
        det_targets = build_det_targets(targets)
        det_loss, _ = model.det_loss(preds, det_targets)
        gps_loss = F.mse_loss(gps_pred, gps)
        
        # Accumulate batch statistics
        batch_size = rgbd.size(0)
        det_total += det_loss.item() * batch_size
        gps_total += gps_loss.item() * batch_size
        sample_count += batch_size
    
    # Calculate averages
    avg_det_loss = det_total / sample_count
    avg_gps_loss = gps_total / sample_count
    
    return avg_det_loss, avg_gps_loss


# ======================================================================
# Main Training Loop
# ======================================================================

def train() -> None:
    """Main training function."""
    # Set random seeds
    set_seeds()
    
    # Print device information
    print(f"Using device: {Config.DEVICE}")
    
    # Build data loaders
    train_loader, val_loader = build_loaders()
    print(f"Dataset: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    
    # Initialize model
    model = YOLO5D().to(Config.DEVICE)
    
    # Create optimizer
    optimizer = create_optimizer(model)
    
    # Create checkpoint directory
    checkpoint_dir = Path("ckpts")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    print(f"\nStarting training for {Config.EPOCHS} epochs...")
    
    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{Config.EPOCHS} ===")
        
        # Train for one epoch
        train_epoch(model, train_loader, optimizer)
        
        # Validate
        det_loss, gps_loss = validate(model, val_loader)
        val_loss = det_loss + gps_loss
        
        print(f"Validation: det_loss={det_loss:.4f}, gps_loss={gps_loss:.6f}, total={val_loss:.4f}")
        
        # Save checkpoint if improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / "yolo5d_best.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ↳ Saved new best checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = checkpoint_dir / "yolo5d_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining completed. Final model saved to {final_path}")


# ======================================================================
# Entry Point
# ======================================================================

if __name__ == "__main__":
    train()
