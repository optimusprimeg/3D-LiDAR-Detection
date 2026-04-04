from setuptools import setup, find_packages
import os
import sys
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Check if CUDA is available
cuda_available = os.environ.get('CUDA_HOME') is not None or \
                 os.path.exists('/usr/local/cuda') or \
                 os.path.exists('/opt/cuda')

ext_modules = []
if cuda_available:
    print("CUDA detected, building with CUDA extensions...")
    ext_modules = [
        CUDAExtension(
            name='pointpillars.ops.voxel_op',
            sources=[
                'pointpillars/ops/voxelization/voxelization.cpp',
                'pointpillars/ops/voxelization/voxelization_cpu.cpp',
                'pointpillars/ops/voxelization/voxelization_cuda.cu',
            ],
            define_macros=[('WITH_CUDA', None)]
        ),
        CUDAExtension(
            name='pointpillars.ops.iou3d_op',
            sources=[
                'pointpillars/ops/iou3d/iou3d.cpp',
                'pointpillars/ops/iou3d/iou3d_kernel.cu',
            ],
            define_macros=[('WITH_CUDA', None)]
        )
    ]
else:
    print("CUDA not detected, building CPU-only version...")
    ext_modules = []

setup(
    name='pointpillars',
    version='0.1',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False
)