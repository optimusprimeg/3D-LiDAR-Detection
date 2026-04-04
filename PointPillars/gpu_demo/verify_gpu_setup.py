#!/workspaces/3D-LiDAR-Detection/PointPillars/venv/bin/python
"""Quick GPU setup verification script."""

import sys
import torch
import numpy as np

print("=" * 60)
print("POINTPILLARS GPU SETUP VERIFICATION")
print("=" * 60)

# 1. PyTorch version
print(f"\n1. PyTorch version: {torch.__version__}")

# 2. CUDA availability
print(f"\n2. CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("   ⚠️  CUDA is NOT available. Make sure:")
    print("   - NVIDIA GPU driver is installed (run: nvidia-smi)")
    print("   - CUDA Toolkit is installed")
    print("   - PyTorch was installed with CUDA support")
    print("   - Reinstall PyTorch with: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

# 3. GPU info
print(f"\n3. CUDA Devices: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"   Device {i}: {torch.cuda.get_device_name(i)}")
    capability = torch.cuda.get_device_capability(i)
    print(f"     Capability: {capability[0]}.{capability[1]}")

# 4. Current device
device = torch.device("cuda:0")
print(f"\n4. Current device: {device}")
print(f"   Device name: {torch.cuda.get_device_name(0)}")

# 5. Memory info
print(f"\n5. GPU Memory:")
allocated = torch.cuda.memory_allocated(0) / 1024**3
reserved = torch.cuda.memory_reserved(0) / 1024**3
total = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"   Allocated: {allocated:.2f} GB")
print(f"   Reserved: {reserved:.2f} GB")
print(f"   Total: {total:.2f} GB")

# 6. Simple tensor test
print(f"\n6. Simple GPU Tensor Test:")
try:
    x = torch.randn(3, 4, 5).to(device)
    y = torch.randn(3, 4, 5).to(device)
    z = torch.matmul(x, y.permute(0, 2, 1))
    print(f"   ✓ Tensor operations work on {device}")
    print(f"   Output shape: {z.shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# 7. Check if PointPillars can load
print(f"\n7. PointPillars Model Test:")
try:
    from pointpillars.model import PointPillars
    model = PointPillars(nclasses=3)
    model.to(device)
    model.eval()
    print(f"   ✓ Model loads and moves to {device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# 8. Check pretrained checkpoint
print(f"\n8. Pretrained Checkpoint Test:")
try:
    ckpt_path = "/workspaces/3D-LiDAR-Detection/PointPillars/pretrained/epoch_160.pth"
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"   ✓ Checkpoint loads successfully on {device}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ GPU Setup is Valid!")
print("=" * 60)
print("\nYou can now run:")
print("  cd /workspaces/3D-LiDAR-Detection")
print("  python PointPillars/gpu_demo/run_gpu_inference.py")
print("\nCheck the output in PointPillars/gpu_demo/demo_inference_output.txt")
