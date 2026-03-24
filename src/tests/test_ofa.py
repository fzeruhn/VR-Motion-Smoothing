import torch
import blackwell_ofa

def main():
    print("Initializing Blackwell OFA Test...")
    
    # Simulate two 8K VR frames sitting on the 5070 Ti (Width: 7680, Height: 4320)
    print("Allocating 8K frames on GPU...")
    frame1 = torch.rand((4320, 7680), device='cuda', dtype=torch.float32)
    frame2 = torch.rand((4320, 7680), device='cuda', dtype=torch.float32)
    
    # Send them to your custom C++ hook!
    print("Calling C++ Hook...")
    motion_vectors = blackwell_ofa.get_vectors(frame1, frame2)
    
    print("--- SUCCESS ---")
    print(f"Input Frame Shape: {frame1.shape}")
    print(f"Motion Vector Shape: {motion_vectors.shape}")
    print(f"Device: {motion_vectors.device}")

if __name__ == "__main__":
    main()