# PointPillars Environment Setup - Complete Guide

## ✓ Environment Setup Complete

The PointPillars 3D LiDAR detection model environment has been successfully configured and the pretrained model is ready to use.

## What Was Done

### 1. **Python Virtual Environment**
- Created isolated Python 3.12 virtual environment at `PointPillars/venv/`
- All dependencies installed separately to avoid conflicts

### 2. **Dependencies Installed**
```
Core ML:
  - torch 2.0+ (with CUDA 11.8 support)
  - torchvision 0.15+
  
Data Processing:
  - numpy >= 1.23.0
  - numba >= 0.57.0
  - opencv-python-headless (for headless environments)
  
Visualization & Utilities:
  - open3d >= 0.16.0
  - tensorboard >= 2.10.0
  - PyYAML >= 6.0
  - tqdm >= 4.64.0
```

### 3. **CUDA Compatibility**
- Since CUDA is not available in the dev container, was made optional
- Created pure Python CPU fallback implementations for:
  - **Hard Voxelization** (`pointpillars/ops/voxel_op.py`) - Converts point clouds to 3D voxels
  - **3D IoU & NMS** (`pointpillars/ops/iou3d_op.py`) - Non-Maximum Suppression and intersection calculations
- Package builds successfully without CUDA (CPU-only mode)

### 4. **Environment Fixes**
- Replaced `opencv-python` with `opencv-python-headless` for headless support
- Made visualization imports optional in `pointpillars/utils/__init__.py`
- Allows model to run in environments without OpenGL/GUI libraries

### 5. **Pretrained Model**
- Successfully loaded: `pretrained/epoch_160.pth`
- Model parameters: **4,834,824**
- Ready for inference on point cloud data

## Quick Start

### Activate Environment
```bash
cd PointPillars
source venv/bin/activate
```

### Run Inference Test
```bash
python test_inference.py
```

### Inference Example
```python
import torch
from pointpillars.model import PointPillars

# Initialize model
model = PointPillars(nclasses=3)  # Pedestrian, Cyclist, Car

# Load pretrained weights
checkpoint = torch.load('pretrained/epoch_160.pth', map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()

# Inference on point cloud (Nx4: x, y, z, intensity)
with torch.no_grad():
    outputs = model(point_cloud.unsqueeze(0))
```

## Dataset Preparation

Before training or evaluating, prepare the KITTI dataset:

```bash
# Download KITTI dataset and organize it as:
# kitti/
#   ├── training/
#   │   ├── calib/
#   │   ├── image_2/
#   │   ├── label_2/
#   │   └── velodyne/
#   └── testing/
#       ├── calib/
#       ├── image_2/
#       └── velodyne/

# Preprocess the dataset
python pre_process_kitti.py --data_root /path/to/kitti
```

## Next Steps

### 1. **Dataset Usage**
```bash
# Preprocess KITTI dataset
python pre_process_kitti.py --data_root /path/to/kitti

# This creates:
# - velodyne_reduced/ (downsampled point clouds)
# - kitti_*infos*.pkl (dataset metadata)
# - kitti_gt_database/ (ground truth database for augmentation)
```

### 2. **Evaluation**
```bash
# Evaluate pretrained model on KITTI val set
python evaluate.py \
    --data_root /path/to/kitti \
    --ckpt pretrained/epoch_160.pth
```

### 3. **Training**
```bash
# Train a new model
python train.py --data_root /path/to/kitti

# Or resume training from checkpoint
python train.py \
    --data_root /path/to/kitti \
    --ckpt /path/to/checkpoint.pth
```

## Troubleshooting

### Import Errors
If you get import errors, make sure the environment is activated:
```bash
source venv/bin/activate
```

### OpenGL/Visualization Issues
The model can run without visualization libraries. If you need visualization:
- On Linux: Install `libgl1` system package
- On macOS/Windows: Visualization should work out of the box

### CUDA Issues
The environment works in CPU-only mode. If CUDA becomes available:
1. Set `CUDA_HOME` environment variable
2. Rebuild with `python setup.py build_ext --inplace`

### Memory Issues
For large datasets, adjust batch size in `python train.py --batch_size 2`

## File Structure
```
PointPillars/
├── venv/                    # Virtual environment (auto-created)
├── pretrained/
│   └── epoch_160.pth       # Pretrained weights
├── pointpillars/
│   ├── model/              # Model architecture
│   ├── dataset/            # Data loaders
│   ├── ops/                # Operations (voxel, NMS, IOU)
│   ├── loss/               # Loss functions
│   └── utils/              # Utilities
├── test_inference.py       # Inference test script
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── pre_process_kitti.py    # Dataset preprocessing
└── requirements.txt        # Dependencies
```

## Performance Notes

Based on repository metrics:
- **mAP (3D BBox):** 73.33% Easy, 62.78% Moderate, 59.63% Hard
- **Classes:** Pedestrian, Cyclist, Car
- **Input:** Point clouds (X, Y, Z, Intensity)
- **Output:** Bounding boxes with class predictions and confidence scores

## Environment Details
- **Python Version:** 3.12.1
- **PyTorch Version:** 2.0+
- **CUDA Support:** Optional (CPU fallback available)
- **Tested on:** Ubuntu 24.04 LTS in dev container

---

**Status:** ✓ Ready to use with dataset and training
