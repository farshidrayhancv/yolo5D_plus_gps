# 5D YOLO Object Detection with GPS Prediction

A PyTorch implementation of YOLOv8 modified for 5-channel input (RGB + depth + thermal) with GPS coordinate prediction.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 1.10+](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-green.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Description

This project extends Ultralytics YOLOv8 for multi-modal object detection by adding:
- 5-channel input support (RGB + depth + thermal)
- GPS coordinate regression (X, Y coordinates)
- Training on Pascal VOC dataset with simulated depth and thermal maps

Perfect for robotics, autonomous vehicles, and multi-sensor fusion applications where object detection must be combined with localization.

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/farshidrayhancv/yolo5D_plus_gps.git
cd 5d-yolo-gps

# Install dependencies
pip install torch torchvision matplotlib tqdm pillow ultralytics
```

### Requirements
- Python 3.8+
- PyTorch 1.10+
- Ultralytics 8.0+
- torchvision
- matplotlib
- tqdm
- Pillow

## 💻 Usage

### Training

```bash
# Basic training
python train_5d.py
```

### Inference

```python
import torch
from Custom5DYOLOModel import Custom5DYOLOModel

# Load model
model = Custom5DYOLOModel()
model.load_state_dict(torch.load("best_5d_yolo_with_gps.pt"))
model.eval()

# Prepare 5-channel input (RGB + depth + thermal)
rgb_img = torch.rand(3, 320, 320)  # Your RGB image
depth_map = torch.rand(1, 320, 320)  # Your depth map
thermal_map = torch.rand(1, 320, 320)  # Your thermal map
input_tensor = torch.cat([rgb_img, depth_map, thermal_map], dim=0).unsqueeze(0)

# Inference
with torch.no_grad():
    outputs = model(input_tensor)

# Get predictions
gps_coords = outputs['gps_coords'].numpy()[0]
print(f"Predicted GPS: [{gps_coords[0]:.4f}, {gps_coords[1]:.4f}]")
```

## 🧠 Model Architecture

The model extends YOLOv8 with:

1. **Modified Input Layer**: First convolutional layer modified to accept 5 channels
2. **Feature Extractor**: YOLOv8 backbone for powerful feature extraction
3. **Object Detection Head**: Standard YOLOv8 detection head
4. **GPS Regression Head**: Custom head that predicts X,Y GPS coordinates

## 📊 Performance

| Model | Input Size | Dataset | GPS Error | Inference Time (CPU) |
|-------|------------|---------|-----------|----------------------|
| 5D-YOLO-n | 320×320 | VOC2012 | ±0.25 | ~150ms |

* GPS predictions trained on fixed coordinates (45, 45)
* Object detection metrics not evaluated as focus was on GPS prediction

## 🔍 Project Structure

```
.
├── train_5d.py          # Main training script
├── best_5d_yolo_with_gps.pt   # Best model weights
├── final_5d_yolo_with_gps.pt  # Final model weights
├── data/                # Dataset directory
├── results/             # Visualization results
└── README.md            # This file
```

## 🔄 Customization

### Adding Real Depth and Thermal Data

Replace the random noise generators with real sensor data:

```python
# In VOCExtended.__getitem__
# Instead of random noise:
depth_map = load_depth_map(depth_path)  # Load your depth map
thermal_map = load_thermal_map(thermal_path)  # Load your thermal map
```

### Training with Different GPS Coordinates

Modify the GPS target values to match your data:

```python
# In VOCExtended.__getitem__
# Instead of fixed coordinates:
gps_coords = torch.tensor([lat, lon], dtype=torch.float32)  # Your GPS data
```

## Implementation Details

This implementation:

1. **Modifies YOLOv8**: 
   - Changes the first convolutional layer to accept 5 channels instead of 3
   - Preserves original weights for RGB channels
   - Initializes new depth and thermal channels with appropriate weights

2. **Creates GPS Head**:
   - Adds a regression head that predicts GPS coordinates
   - Uses adaptive pooling to handle different feature map sizes
   - Implements a custom loss function for GPS coordinate prediction

3. **Data Processing**:
   - Automatically downloads Pascal VOC dataset
   - Generates synthetic depth and thermal maps
   - Sets fixed GPS coordinates (45, 45) for training
   - Visualizes results with ground truth boxes and GPS predictions
  
4. **PipeLine**:
   
   flowchart TD
    %% ================== DATA PIPELINE ==================
    A[Pascal-VOC image\n+ synthetic Depth + Thermal + GPS] --> B[DataLoader<br/>(batch & collate)]

    %% ==================== MODEL ========================
    subgraph YOLO5D
        direction LR
        B --> C[RGBD2RGB<br/>4→3 1×1 conv]
        C --> D[YOLOv8-n Backbone]
        B -->|Thermal| E[ThermalProcessor<br/>embed + upsample]
        E --> F{{Layer 6<br/>fusion hook}}
        F --> D
        D --> G[Detection head]      %% internal YOLO head
        D --> H[GPSHead<br/>lazy MLP]
    end

    %% ================ TRAINING LOSSES ==================
    G --> I[Detection Loss<br/>(v8DetectionLoss)]
    H --> J[GPS Loss<br/>(MSE)]
    I & J --> K[Total Loss = I + λ·J]
    K --> L[Adam Optimiser<br/>LR_backbone 1e-4<br/>LR_new 1e-3]
    L --> M[(Weights update)]
    M --> Q[Checkpoint<br/>yolo5d_*.pt]

    %% ================== INFERENCE ======================
    G -. raw preds .-> N[Post-processing<br/>Ultralytics new_results]
    N --> O[Final Detections]
    H --> P[GPS Output]

    %% ================ SCRIPT ENTRY =====================
    Q -.-> R[train()/val()<br/>main()]

## 🔄 TODO

- [ ] Add support for real depth and thermal datasets
- [ ] Implement loss weighting between detection and GPS tasks
- [ ] Add evaluation metrics for multi-modal detection
- [ ] Optimize for edge devices and real-time inference
- [ ] Create TensorRT and ONNX export options

## 📄 License

This project is released under the MIT License.

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the base object detection framework
- [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) for the dataset
