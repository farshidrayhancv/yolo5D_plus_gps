"""
Configuration parameters for 5D YOLOv8 + GPS model
"""
import torch

# Model parameters
IMG_SIZE = 320            # Square input size
THERMAL_SIZE = 96         # Thermal input size
THERMAL_WEIGHT = 0.1      # Weight for thermal feature fusion
YOLO_WEIGHTS = "yolov8n.pt"  # Base YOLOv8 weights

# Training parameters
SEED = 42                 # Random seed for reproducibility
BATCH_SIZE = 8            # Training batch size
EPOCHS = 30               # Number of training epochs
LR_BACKBONE = 1e-4        # Learning rate for pre-trained backbone
LR_NEW = 1e-3             # Learning rate for new layers
LAMBDA_GPS = 1.0          # GPS loss weight
VAL_SPLIT = 0.1           # Validation split ratio
WORKERS = 4               # Number of data loader workers

# Data parameters
DATA_ROOT = "data"        # Dataset root directory
DATASET_YEAR = "2012"     # VOC dataset year

# Inference parameters
CONF_THRESH = 0.25        # Confidence threshold for NMS
IOU_THRESH = 0.7          # IoU threshold for NMS

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")