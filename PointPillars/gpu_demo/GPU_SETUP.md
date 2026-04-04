# GPU Demo Setup

This folder (`gpu_demo/`) is configured to run PointPillars inference on GPU using CUDA+Nvidia.

## Files

- `run_gpu_inference.py` - Main demo script with GPU support
  - Auto-detects CUDA/GPU availability
  - Falls back to CPU if GPU is not available
  - Pass `--no-cuda` to force CPU-only mode

## Prerequisites

Before running, ensure your environment has:

1. **NVIDIA GPU Driver** installed
2. **CUDA Toolkit** (version 11.x or 12.x recommended)
3. **cuDNN** (optional but recommended for better performance)
4. **PyTorch with CUDA support** (in the venv)

## Setup Steps

### Step 1: Switch to GPU-capable CUDA environment

```bash
# If using conda or manual CUDA installation, activate it
# Example (adjust version to match your CUDA toolkit):
# export PATH=/usr/local/cuda-11.8/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

### Step 2: Reinstall PyTorch with CUDA support

```bash
cd /workspaces/3D-LiDAR-Detection
source PointPillars/venv/bin/activate

# Remove old CPU-only torch
pip uninstall torch torchvision torchaudio -y

# Install PyTorch with CUDA 12.1 (adjust version as needed)
# Visit https://pytorch.org to get the correct command for your CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Verify CUDA support

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

## Running the Demo

```bash
# From the repo root
cd /workspaces/3D-LiDAR-Detection

# Run with auto GPU detection (will use GPU if available)
python PointPillars/gpu_demo/run_gpu_inference.py

# Run with specific output locations
python PointPillars/gpu_demo/run_gpu_inference.py \
  --output /path/to/output.txt \
  --visual-output-dir /path/to/visuals/

# Force CPU-only mode (fallback)
python PointPillars/gpu_demo/run_gpu_inference.py --no-cuda
```

## Expected Output

When CUDA is available and working:

1. **demo_inference_output.txt** - Text report showing:
   - Device used (GPU name or CPU)
   - CUDA capability info
   - Model output structure (should show predicted boxes as dicts, not empty tuples)
   - Scene object summaries
   
2. **demo_visuals/** directory containing:
   - `val_000134_gt.png` - Ground truth overlay
   - `val_000134_pred.png` - **Predicted boxes overlay** (should have actual detections if GPU is working)
   - `test_000002_pred.png` - Prediction overlay for test sample

## Troubleshooting

**Issue: CUDA not detected**
- Run `nvidia-smi` to verify GPU driver is installed
- Check CUDA Toolkit installation: `nvcc --version`
- Ensure PyTorch was installed with CUDA support: `pip show torch` (look for cuda in location)

**Issue: Model predictions still empty**
- This could mean the checkpoint requires specific training conditions
- Try with GPU first; if still empty, the checkpoint may need retraining on your system
- Check the report for actual confidence score statistics

**Issue: Out of memory errors**
- Reduce batch size or use smaller point clouds
- Modify `max_voxels` in the model config

## Comparison

| Environment | Max Confidence | Predicted Boxes | Status |
|-------------|---|---|---|
| CPU-only | 0.0677 | 0 after filter | Unusable |
| GPU (CUDA) | Expected: >0.3 | Expected: >0 | Target |

Once GPU is working, the predicted overlays should show actual bounding boxes on the camera images.
