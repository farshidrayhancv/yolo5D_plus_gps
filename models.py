"""
Model components for 5D YOLOv8 + GPS
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from typing import Dict, List, Tuple, Union

from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss

import config as cfg


class RGBD2RGB(nn.Module):
    """Learnable adapter to convert 4-channel RGB-D to 3-channel input."""
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
    """Process and upsample thermal input to match feature dimensions."""
    def __init__(self, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1)
        )
        
    def forward(self, thermal: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # Upsample to target dimensions
        thermal_resized = F.interpolate(
            thermal, size=(height, width), 
            mode="bilinear", align_corners=False
        )
        # Process through convolution network
        return self.net(thermal_resized)


class GPSHead(nn.Module):
    """GPS regression head with dynamic input dimension."""
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = None  # Will be initialized on first forward pass
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, 2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class YOLO5D(nn.Module):
    """YOLOv8 model extended for RGB-D-Thermal input and GPS regression."""
    def __init__(self, weights=None):
        super().__init__()
        # Use specified weights or default from config
        weights = weights or cfg.YOLO_WEIGHTS
        
        # Load base YOLO model
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
        """Probe mid-level backbone features to determine channel count."""
        # Create dummy input
        dummy = torch.zeros(1, 4, cfg.IMG_SIZE, cfg.IMG_SIZE)
        
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
            return (x + cfg.THERMAL_WEIGHT * thermal_features,)
        
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
        """Set model to evaluation mode while keeping backbone in training mode."""
        super().eval()                 # Sets all modules to eval
        self.backbone.train(True)      # Keep YOLO in train-mode
        return self
    
    def to(self, *args, **kwargs):
        """Move model to device and handle detection loss tensors."""
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
                conf: float = None, 
                iou: float = None):
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
        # Use config values if not specified
        conf = conf if conf is not None else cfg.CONF_THRESH
        iou = iou if iou is not None else cfg.IOU_THRESH
        
        # Set to eval mode
        self.eval()
        
        # Forward pass
        preds, gps = self(rgbd, thermal)
        
        # Post-process detections
        results = self.backbone.model[-1].new_results(
            preds, rgbd,
            proto=None,
            imgsz=cfg.IMG_SIZE,
            conf=conf, iou=iou
        )
        
        return results, gps