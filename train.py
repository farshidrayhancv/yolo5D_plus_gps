"""
Training script for 5D YOLOv8 + GPS

Trains model on Pascal VOC dataset with synthetic depth, thermal, and GPS data.
"""
import argparse
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

import config as cfg
from models import YOLO5D
from dataset import get_dataloaders, build_det_targets
from utils import set_seeds, create_optimizer, save_checkpoint


def train_epoch(model, train_loader, optimizer, epoch):
    """Train model for one epoch."""
    model.train()
    det_total = gps_total = sample_count = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS}")
    
    for rgbd, thermal, targets, gps in progress_bar:
        # Move data to device
        rgbd = rgbd.to(cfg.DEVICE, non_blocking=True)
        thermal = thermal.to(cfg.DEVICE, non_blocking=True)
        gps = gps.to(cfg.DEVICE, non_blocking=True)
        
        # Forward pass
        preds, gps_pred = model(rgbd, thermal)
        
        # Build detection targets
        det_targets = build_det_targets(targets)
        
        # Calculate losses
        det_loss, _ = model.det_loss(preds, det_targets)
        gps_loss = F.mse_loss(gps_pred, gps)
        total_loss = det_loss + cfg.LAMBDA_GPS * gps_loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Accumulate batch statistics
        batch_size = rgbd.size(0)
        det_total += det_loss.item() * batch_size
        gps_total += gps_loss.item() * batch_size
        sample_count += batch_size
        
        # Update progress bar
        progress_bar.set_postfix(
            det=f"{det_loss.item():.4f}", 
            gps=f"{gps_loss.item():.6f}"
        )
    
    # Calculate averages
    avg_det_loss = det_total / sample_count
    avg_gps_loss = gps_total / sample_count
    
    return avg_det_loss, avg_gps_loss


@torch.no_grad()
def validate(model, val_loader):
    """Validate model on validation dataset."""
    model.eval()
    det_total = gps_total = sample_count = 0
    
    for rgbd, thermal, targets, gps in val_loader:
        # Move data to device
        rgbd = rgbd.to(cfg.DEVICE, non_blocking=True)
        thermal = thermal.to(cfg.DEVICE, non_blocking=True)
        gps = gps.to(cfg.DEVICE, non_blocking=True)
        
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


def main():
    """Main training function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train 5D YOLOv8 + GPS")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path or 'best'")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    args = parser.parse_args()
    
    # Override config from command-line arguments
    if args.batch_size is not None:
        cfg.BATCH_SIZE = args.batch_size
    if args.epochs is not None:
        cfg.EPOCHS = args.epochs
    
    # Set random seeds
    set_seeds()
    
    # Create checkpoint directory
    checkpoint_dir = Path("ckpts")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Print device information
    print(f"Using device: {cfg.DEVICE}")
    
    # Build data loaders
    train_loader, val_loader = get_dataloaders()
    print(f"Dataset: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    
    # Initialize model
    model = YOLO5D().to(cfg.DEVICE)
    
    # Create optimizer
    optimizer = create_optimizer(model)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        checkpoint_path = checkpoint_dir / "yolo5d_best.pt" if args.resume == "best" else args.resume
        if os.path.isfile(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_val_loss = checkpoint.get("loss", float('inf'))
            print(f"Starting from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
    
    # Training loop
    print(f"\nStarting training for {cfg.EPOCHS} epochs...")
    
    for epoch in range(start_epoch, cfg.EPOCHS):
        # Train for one epoch
        train_det_loss, train_gps_loss = train_epoch(model, train_loader, optimizer, epoch + 1)
        
        # Validate
        val_det_loss, val_gps_loss = validate(model, val_loader)
        val_loss = val_det_loss + val_gps_loss
        
        print(f"Epoch {epoch+1}/{cfg.EPOCHS}:")
        print(f"  Train: det_loss={train_det_loss:.4f}, gps_loss={train_gps_loss:.6f}")
        print(f"  Val: det_loss={val_det_loss:.4f}, gps_loss={val_gps_loss:.6f}, total={val_loss:.4f}")
        
        # Save regular checkpoint
        checkpoint_path = save_checkpoint(
            model=model,
            epoch=epoch + 1,
            optimizer=optimizer,
            loss=val_loss,
            checkpoint_dir=checkpoint_dir
        )
        
        # Save best checkpoint if improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = save_checkpoint(
                model=model,
                epoch=epoch + 1,
                optimizer=optimizer,
                loss=val_loss,
                is_best=True,
                checkpoint_dir=checkpoint_dir
            )
            print(f"  ↳ Saved new best checkpoint: {best_checkpoint_path}")
    
    print(f"\nTraining completed. Final model saved.")


if __name__ == "__main__":
    main()