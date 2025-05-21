"""
Utility functions for 5D YOLOv8 + GPS
"""
import os
import random
import torch
import numpy as np
import torch.optim as optim
from pathlib import Path
from typing import Dict

import config as cfg
from models import YOLO5D


def set_seeds(seed=None):
    """Set random seeds for reproducibility."""
    seed = seed if seed is not None else cfg.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def create_optimizer(model: YOLO5D) -> optim.Optimizer:
    """Create optimizer with different learning rates for different parts."""
    backbone_params, new_params = [], []
    
    # Separate parameters into groups
    for name, param in model.named_parameters():
        if any(k in name for k in ['adapt', 't_fuse', 'gps_head']):
            new_params.append(param)
        else:
            backbone_params.append(param)
    
    # Create optimizer with different learning rates
    return optim.Adam([
        {"params": backbone_params, "lr": cfg.LR_BACKBONE},
        {"params": new_params, "lr": cfg.LR_NEW}
    ])


def save_checkpoint(model, epoch, optimizer, loss, is_best=False, checkpoint_dir="ckpts"):
    """Save model checkpoint."""
    # Convert to Path object if string
    if isinstance(checkpoint_dir, str):
        checkpoint_dir = Path(checkpoint_dir)
    
    # Create directory if it doesn't exist
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Determine checkpoint path
    if is_best:
        checkpoint_path = checkpoint_dir / "yolo5d_best.pt"
    else:
        checkpoint_path = checkpoint_dir / f"yolo5d_epoch{epoch:03d}.pt"
    
    # Create checkpoint
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    return checkpoint_path


def load_checkpoint(model, checkpoint_path, optimizer=None):
    """Load model checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE)
    
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint