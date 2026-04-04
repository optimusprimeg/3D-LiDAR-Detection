#!/bin/bash
# Quick commands for PointPillars workflow

# ============================================================
# ENVIRONMENT SETUP (ALREADY DONE)
# ============================================================

# Activate environment (DO THIS FIRST!)
source PointPillars/venv/bin/activate

# ============================================================
# VERIFICATION & TESTING
# ============================================================

# Test that model loads and inference works
cd PointPillars && python test_inference.py

# Run the original test.py (requires point cloud files)
python test.py --pc_path <path_to_point_cloud.bin> --ckpt pretrained/epoch_160.pth

# ============================================================
# DATASET PREPARATION
# ============================================================

# Download KITTI dataset first, then organize as:
# kitti/
#   ├── training/velodyne/
#   ├── training/calib/
#   ├── training/image_2/
#   ├── training/label_2/
#   └── testing/...

# Preprocess the dataset (creates velodyne_reduced, infos.pkl, etc.)
python pre_process_kitti.py --data_root /path/to/kitti

# ============================================================
# EVALUATION
# ============================================================

# Evaluate pretrained model
python evaluate.py --data_root /path/to/kitti --ckpt pretrained/epoch_160.pth

# ============================================================
# TRAINING
# ============================================================

# Train from scratch
python train.py --data_root /path/to/kitti --batch_size 4

# Resume from checkpoint
python train.py --data_root /path/to/kitti --ckpt checkpoints/latest.pth

# ============================================================
# USEFUL FLAGS
# ============================================================

# Common training args:
# --batch_size        : Batch size (default: 4)
# --num_workers      : DataLoader workers (default: 4)
# --lr               : Learning rate (default: 0.001)
# --ckpt             : Checkpoint to resume from
# --no_cuda          : Use CPU only

# Example:
python train.py --data_root /path/to/kitti --batch_size 2 --lr 0.0005 --no_cuda

# ============================================================
# ENVIRONMENT INFO
# ============================================================

# Show installed packages
pip list

# Show PyTorch info
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Deactivate environment
deactivate

# ============================================================
# TROUBLESHOOTING
# ============================================================

# If imports fail, ensure environment is activated:
source PointPillars/venv/bin/activate

# If you have CUDA issues, you can still run in CPU mode:
python train.py --data_root /path/to/kitti --no_cuda --batch_size 1

# Check available disk space before preprocessing (needs ~50GB for KITTI):
df -h

# Monitor training progress:
tensorboard --logdir <output_dir>/summary --port 6006
