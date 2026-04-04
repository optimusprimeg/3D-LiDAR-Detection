#!/bin/bash
# Quick GPU setup and verification script for PointPillars

set -e

echo "=========================================="
echo "PointPillars GPU Setup Quick Start"
echo "=========================================="

REPO_ROOT="/workspaces/3D-LiDAR-Detection"
cd "$REPO_ROOT"

echo ""
echo "Step 1: Activate venv"
source PointPillars/venv/bin/activate

echo ""
echo "Step 2: Verify NVIDIA GPU is detected"
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA GPU driver not found. Install NVIDIA driver first."
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo ""
echo "Step 3: Check CUDA Toolkit"
if ! command -v nvcc &> /dev/null; then
    echo "⚠️  CUDA Toolkit (nvcc) not in PATH. Activation might be needed."
    echo "   Try: export PATH=/usr/local/cuda-XX/bin:\$PATH"
else
    nvcc --version
fi

echo ""
echo "Step 4: Verify PyTorch CUDA support"
python -c "
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA Available: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA Version: {torch.version.cuda}')
else:
    print('❌ CUDA not detected. Reinstall PyTorch with CUDA support:')
    print('   pip install torch --index-url https://download.pytorch.org/whl/cu121')
    exit(1)
"

echo ""
echo "Step 5: Run GPU Verification"
python PointPillars/gpu_demo/verify_gpu_setup.py

echo ""
echo "Step 6: Ready to run GPU demo!"
echo "Command: python PointPillars/gpu_demo/run_gpu_inference.py"
