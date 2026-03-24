import torch
import torch.nn.functional as F

class VFMotionSmoother:
    def __init__(self, width, height, target_fps=90):
        self.width = width
        self.height = height
        self.target_fps = target_fps
        
        # Pre-compute the base normalized pixel grid [-1, 1]
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, dtype=torch.float32, device='cuda'),
            torch.linspace(-1, 1, width, dtype=torch.float32, device='cuda'),
            indexing='ij'
        )
        # grid shape: (1, height, width, 2)
        self.base_grid = torch.stack((x, y), dim=-1).unsqueeze(0)

    # Warps a single color frame (3, H, W) using the scaled flow vectors.
    def warp_frame(self, frame_color, flow, t_scale):

        flow_scaled = flow.clone()
        flow_scaled[..., 0] = (flow_scaled[..., 0] * t_scale) / (self.width / 2.0)
        flow_scaled[..., 1] = (flow_scaled[..., 1] * t_scale) / (self.height / 2.0)
        
        warp_grid = self.base_grid + flow_scaled.unsqueeze(0)
        
        # Frame is (3, H, W). We add batch dim to make it (1, 3, H, W) for grid_sample
        frame_bchw = frame_color.unsqueeze(0)
        
        # Warp!
        warped = F.grid_sample(frame_bchw, warp_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        # Strip batch dim to return (3, H, W)
        return warped.squeeze(0) 
    
    # Estimates disocclusions (holes) by warping the depth map and detecting 
    # severe stretching caused by diverging motion vectors, plus edge boundaries
    def get_hole_mask(self, depth_map, flow, t):
        # 1. Warp the depth map (needs a channel dim: 1, H, W)
        warped_depth = self.warp_frame(depth_map.unsqueeze(0), flow, t_scale=(1 - t))
        
        # 2. Calculate spatial gradients to find "stretched" tears in the Z-buffer
        dx = torch.abs(warped_depth[:, :, 1:] - warped_depth[:, :, :-1])
        dy = torch.abs(warped_depth[:, 1:, :] - warped_depth[:, :-1, :])
        
        # Pad back to original (H, W) size
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        
        depth_edges = dx + dy
        
        # 3. Threshold to binary mask (Assumes depth is normalized 0.0 - 1.0)
        # You may need to tune '0.05' based on your engine's near/far clipping planes
        disocclusion_mask = (depth_edges > 0.05).to(torch.float32)
        
        # 4. Find out-of-bounds pixels (screen edges) by warping a solid block of 1s
        ones = torch.ones_like(depth_map).unsqueeze(0)
        warped_ones = self.warp_frame(ones, flow, t_scale=(1 - t))
        edge_mask = (warped_ones < 0.99).to(torch.float32)
        
        # 5. Combine and return as (1, H, W)
        final_mask = torch.clamp(disocclusion_mask + edge_mask, 0.0, 1.0)
        return final_mask

    # Dynamically outputs the necessary frames to hit 90 FPS.
    def generate_frames(self, frame_prev, frame_curr, depth_map, motion_vectors, input_fps):
            
        # Decode OFA Fixed-Point Vectors to Float32 Pixel Offsets
        flow_f32 = motion_vectors.to(torch.float32) / 16.0 
        
        ratio = round(self.target_fps / input_fps)
        synthetic_frames = []
        hole_masks = []
        
        for step in range(1, ratio):
            t = step / ratio
            
            # Warp color frames (ensure they are float for grid_sample)
            warp_f0 = self.warp_frame(frame_prev.float(), flow_f32, t_scale=-t)
            warp_f1 = self.warp_frame(frame_curr.float(), flow_f32, t_scale=(1 - t))
            
            # Assume depth_map is normalized 0.0 (far) to 1.0 (close)
            # We warp the depth map forward so it aligns with the synthetic frame
            warp_d0 = self.warp_frame(depth_map.unsqueeze(0), flow_f32, t_scale=-t)
            warp_d1 = self.warp_frame(depth_map.unsqueeze(0), flow_f32, t_scale=(1 - t))
            
            # Create dynamic weights: High depth (closer) + Time proximity wins
            weight_f0 = torch.exp(warp_d0 * 5.0) * (1 - t)
            weight_f1 = torch.exp(warp_d1 * 5.0) * t
            
            # Normalize the weights so they sum to 1.0
            sum_weights = weight_f0 + weight_f1 + 1e-6 # add epsilon to avoid div by zero
            w0 = weight_f0 / sum_weights
            w1 = weight_f1 / sum_weights
            
            # Blend using the occlusion-aware weights
            blended_frame = (warp_f0 * w0 + warp_f1 * w1)

            # Calculate the real hole mask for this timestep
            current_mask = self.get_hole_mask(depth_map, flow_f32, t)
            
            synthetic_frames.append(blended_frame.to(torch.uint8))
            hole_masks.append(current_mask)
            
        return synthetic_frames, hole_masks