"""
Inference script for 5D YOLOv8 + GPS model
"""
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import config as cfg
from models import YOLO5D
from dataset import VOCExtended
from utils import load_checkpoint


def load_image(image_path, depth_path=None, thermal_path=None):
    """Load and preprocess image data for inference."""
    # Load RGB image
    with Image.open(image_path) as img:
        img = img.resize((cfg.IMG_SIZE, cfg.IMG_SIZE))
        rgb = torch.from_numpy(np.array(img)).float() / 255.0
        rgb = rgb.permute(2, 0, 1)[:3]  # Take only first 3 channels
    
    # Load or generate depth
    if depth_path:
        with Image.open(depth_path) as depth_img:
            depth_img = depth_img.resize((cfg.IMG_SIZE, cfg.IMG_SIZE))
            depth = torch.from_numpy(np.array(depth_img)).float() / 255.0
            if depth.dim() == 3:
                depth = depth[:, :, 0]
            depth = depth.unsqueeze(0)
    else:
        depth = torch.rand(1, cfg.IMG_SIZE, cfg.IMG_SIZE)
    
    # Load or generate thermal
    if thermal_path:
        with Image.open(thermal_path) as thermal_img:
            thermal_img = thermal_img.resize((cfg.THERMAL_SIZE, cfg.THERMAL_SIZE))
            thermal = torch.from_numpy(np.array(thermal_img)).float() / 255.0
            if thermal.dim() == 3:
                thermal = thermal[:, :, 0]
            thermal = thermal.unsqueeze(0)
    else:
        thermal = torch.rand(1, cfg.THERMAL_SIZE, cfg.THERMAL_SIZE)
    
    # Combine RGB and depth
    rgbd = torch.cat([rgb, depth], dim=0)
    
    # Add batch dimension
    rgbd = rgbd.unsqueeze(0)
    thermal = thermal.unsqueeze(0)
    
    return rgbd, thermal


def visualize_results(image_path, results, gps, output_path=None):
    """Visualize detection and GPS results."""
    # Load original image
    img = Image.open(image_path)
    img = img.resize((cfg.IMG_SIZE, cfg.IMG_SIZE))
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Show image with bounding boxes
    ax1.imshow(np.array(img))
    ax1.set_title("Object Detection")
    ax1.axis('off')
    
    # Draw bounding boxes and labels
    result = results[0]  # Get first batch result
    
    if len(result.boxes) > 0:
        # Convert normalized coordinates to pixel coordinates
        boxes = result.boxes.xyxy.cpu().numpy()
        
        # Get class indices and confidence scores
        cls_idx = result.boxes.cls.cpu().numpy().astype(int)
        conf = result.boxes.conf.cpu().numpy()
        
        # Draw each box
        for box, cls_id, conf_score in zip(boxes, cls_idx, conf):
            x1, y1, x2, y2 = box
            
            # Draw rectangle
            rect = plt.Rectangle(
                (x1, y1), x2-x1, y2-y1, 
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax1.add_patch(rect)
            
            # Add label
            cls_name = VOCExtended.CLASSES[cls_id] if cls_id < len(VOCExtended.CLASSES) else f"Class {cls_id}"
            label = f"{cls_name}: {conf_score:.2f}"
            ax1.text(
                x1, y1-5, label, 
                bbox=dict(facecolor='white', alpha=0.7),
                fontsize=8
            )
    
    # Show GPS coordinates on a simple map
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title("GPS Coordinates")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.grid(True)
    
    # Extract GPS coordinates
    lat, lon = gps[0].cpu().numpy()
    
    # Plot GPS point
    ax2.scatter(lon, lat, c='red', s=100, marker='x')
    ax2.text(lon+0.02, lat, f"({lat:.4f}, {lon:.4f})", fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()


def main():
    """Main inference function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="5D YOLOv8 + GPS Inference")
    parser.add_argument("--model", type=str, default="ckpts/yolo5d_best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--image", type=str, required=True,
                       help="Path to input RGB image")
    parser.add_argument("--depth", type=str, default=None,
                       help="Path to depth image (optional)")
    parser.add_argument("--thermal", type=str, default=None,
                       help="Path to thermal image (optional)")
    parser.add_argument("--conf", type=float, default=cfg.CONF_THRESH,
                       help="Confidence threshold for NMS")
    parser.add_argument("--iou", type=float, default=cfg.IOU_THRESH,
                       help="IoU threshold for NMS")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save visualization")
    args = parser.parse_args()
    
    # Load model
    model = YOLO5D().to(cfg.DEVICE)
    load_checkpoint(model, args.model)
    model.eval()
    
    # Load image
    rgbd, thermal = load_image(args.image, args.depth, args.thermal)
    
    # Move to device
    rgbd = rgbd.to(cfg.DEVICE)
    thermal = thermal.to(cfg.DEVICE)
    
    # Run inference
    with torch.no_grad():
        results, gps = model.predict(rgbd, thermal, conf=args.conf, iou=args.iou)
    
    # Print results
    print(f"Detected objects: {len(results[0].boxes)}")
    if len(results[0].boxes) > 0:
        print(f"Bounding boxes: {results[0].boxes.xyxy}")
        print(f"Classes: {results[0].boxes.cls}")
        print(f"Confidence: {results[0].boxes.conf}")
    print(f"GPS coordinates: {gps.squeeze().tolist()}")
    
    # Visualize results
    visualize_results(args.image, results, gps, args.output)


if __name__ == "__main__":
    main()