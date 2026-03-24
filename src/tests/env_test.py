import torch
import tensorrt as trt
import cupy as cp
import pycuda.driver as cuda_driver
import cv2
import sys
import numpy as np
import blackwell_ofa

# 1. Initialize CUDA for PyCUDA
try:
    cuda_driver.init()
    pycuda_status = f"Success ({cuda_driver.Device.count()} device(s) found)"
except Exception as e:
    pycuda_status = f"FAILED: {e}"

print(f"\n{'='*20} Environment Health Check {'='*20}")
print(f"Python Version:   {sys.version.split()[0]}")
print(f"GPU Model:        {torch.cuda.get_device_name(0)}")
print(f"PyTorch CUDA:     {torch.version.cuda} (Active: {torch.cuda.is_available()})")
print(f"TensorRT Version: {trt.__version__}")
print(f"CuPy Device:      {cp.array([1]).device}")
print(f"PyCUDA Status:    {pycuda_status}")

print(f"\n{'='*20} OpenCV & Hardware Acceleration {'='*20}")
print(f"OpenCV Version:   {cv2.__version__}")

# Check for CUDA Module
cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
print(f"OpenCV CUDA:      {'YES' if cuda_count > 0 else 'NO'}")

# Check for cuDNN Integration
build_info = cv2.getBuildInformation()
has_cudnn = "cuDNN: YES" in build_info
print(f"OpenCV cuDNN:     {'YES' if has_cudnn else 'NO (Check CMake paths if needed later)'}")

print(f"\n{'='*20} OpenCV GPU Architevture Check {'='*20}")
build_info = cv2.getBuildInformation()

# Filter the massive output to just the lines we care about
for line in build_info.split('\n'):
    if 'NVIDIA GPU arch' in line or 'NVIDIA PTX archs' in line or 'cuDNN' in line or 'CUDA' in line:
        print(line.strip())

print(f"\n{'='*20} OFA Test {'='*20}")
print("Starting VR Engine Test...")
    
# 1. Initialize the Hardware ONCE (Pass your 8K dimensions)
# Width = 7680, Height = 4320
ofa_engine = blackwell_ofa.Engine(7680, 4320)
    
# 2. Allocate VRAM buffers
frame1 = torch.rand((4320, 7680), device='cuda', dtype=torch.float32)
frame2 = torch.rand((4320, 7680), device='cuda', dtype=torch.float32)
    
# 3. Stream frames into the silicon
print("Sending VRAM pointers to Blackwell Silicon...")
motion_vectors = ofa_engine.calc(frame1, frame2)
print(f"SUCCESS: Motion Vector Shape: {motion_vectors.shape} on {motion_vectors.device}")