# Add this to the existing argument parser in train.py:
parser.add_argument("--dataset-path", type=str, default=None,
                    help="Path to real multi-modal dataset (if None, uses synthetic data)")


# Then modify the dataloader creation part:

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
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Path to real multi-modal dataset (if None, uses synthetic data)")
    args = parser.parse_args()
    
    # Override config from command-line arguments
    if args.batch_size is not None:
        cfg.BATCH_SIZE = args.batch_size
    if args.epochs is not None:
        cfg.EPOCHS = args.epochs
    
    # Set random seeds
    set_seeds()
    
    # Print device information
    print(f"Using device: {cfg.DEVICE}")
    
    # Build data loaders - pass dataset path if specified
    train_loader, val_loader = get_dataloaders(args.dataset_path)
    
    print(f"Dataset: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    