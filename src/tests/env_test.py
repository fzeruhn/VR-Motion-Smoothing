import torch
import tensorrt as trt
import cupy as cp
import pycuda.driver as cuda_driver
import cv2
import sys
import numpy as np

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

# 2. Hardware Stress Test: The Nvidia Optical Flow (OFA) Engine
print(f"\n{'='*20} Blackwell OFA Hardware Test {'='*20}")

width, height = 1024, 1024
roiData = None  
preset = 5      # NV_OF_PERF_LEVEL_SLOW (Highest quality)

# Allocate GPU Memory for the frames (directly in VRAM)
gpu_frame1 = cv2.cuda_GpuMat(height, width, cv2.CV_8UC3)
gpu_frame2 = cv2.cuda_GpuMat(height, width, cv2.CV_8UC3)

# Fill them with zeros so the hardware doesn't process random garbage memory
gpu_frame1.upload(np.zeros((height, width, 3), dtype=np.uint8))
gpu_frame2.upload(np.zeros((height, width, 3), dtype=np.uint8))

print("Hunting for Blackwell-compatible OFA Grid Size (4x4, 2x2, 1x1)...")
ofa_success = False

# Grid size mapping: 1 usually means 4x4, 2 means 2x2, 4 means 1x1
for g_size in [1, 2, 4]:
    try:
        # THE FIX: Strictly positional arguments. No keywords allowed!
        nvof = cv2.cuda.NvidiaOpticalFlow_2_0_create((width, height), None, 5, g_size)
        
        # Run the calculation
        flow = nvof.calc(gpu_frame1, gpu_frame2, None)
        
        print(f"\nOFA Status:       SUCCESS (Hardware Session Active!)")
        print(f"Accepted Grid:    {g_size}")
        print(f"OFA Output Size:  {flow.size()}")
        
        # Important: Release the hardware session to prevent VRAM leaks
        nvof.collectGarbage()
        ofa_success = True
        break  # Exit the loop since we found the working configuration
        
    except Exception as e:
        # Clean up the error output so it's easier to read
        error_msg = str(e).split('\n')[0]
        print(f" -> Grid Size {g_size} rejected: {error_msg}")

if not ofa_success:
    print(f"\nOFA Status:       CRITICAL FAILURE")
    print("Error Detail:     All OFA grid sizes failed.")

print(f"\n{'='*66}")