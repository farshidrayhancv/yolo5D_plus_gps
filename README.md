# 5D YOLOv8 + GPS — Multi-Modal Object Detection and Localization

A PyTorch implementation of Ultralytics **YOLOv8** extended to consume  
**RGB + Depth + Thermal** (5-channel) input and regress a global **GPS (lat, lon)**  
for every frame.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)  
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)](https://pytorch.org/)  
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](LICENSE)

## 📋 Overview

* **5-channel input processing:**
  * RGB (3 channels) + Depth (1 channel) → 4-channel tensor
  * Thermal (1 channel) → independent processing path
* **Modality fusion architecture:**
  * RGB-D → 1×1 conv adapter → YOLOv8 backbone
  * Thermal → CNN → injected at backbone layer-6 via hook
* **Dual-output model:**
  * Standard YOLO detection head (boxes, classes)
  * GPS regression head (2D coordinates from feature map)
* **Joint training:**
  * Ultralytics v8 detection loss + MSE GPS loss
  * Different learning rates for new layers vs backbone

Ideal for robotics, autonomous driving, or any scenario where object
detection and coarse localization must be learned from multiple sensors.

## 🛠 Installation

```bash
git clone https://github.com/yourusername/yolo5D_plus_gps.git
cd yolo5D_plus_gps
```

## Core Dependencies
```bash
pip install torch torchvision matplotlib tqdm pillow ultralytics==8.3.140
```

**Requirements:** Python 3.8+, PyTorch 2.1+,
