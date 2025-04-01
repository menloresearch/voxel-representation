import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
# import plotly.graph_objs as go # Not used in the provided script snippet
# Assuming Voxelizer is correctly importable from your project structure
from src.openscene.voxelizer import Voxelizer # Make sure this path is correct

def get_scannet_color_map():
    """
    Define a color map for ScanNet dataset object classes. Returns RGB tuples (0-255).
    """
    # Using integer tuples directly as colors are often handled as uint8
    return {
        1: (174, 199, 232),  # wall
        2: (152, 223, 138),  # floor
        3: (31, 119, 180),   # cabinet
        4: (255, 187, 120),  # bed
        5: (188, 189, 34),   # chair
        6: (140, 86, 75),    # sofa
        7: (255, 152, 150),  # table
        8: (214, 39, 40),    # door
        9: (197, 176, 213),  # window
        10: (148, 103, 189), # bookshelf
        11: (196, 156, 148), # picture
        12: (23, 190, 207),  # counter
        14: (247, 182, 210), # desk
        16: (219, 219, 141), # curtain
        24: (255, 127, 14),  # refrigerator
        28: (158, 218, 229), # shower curtain
        33: (44, 160, 44),   # toilet
        34: (112, 128, 144), # sink
        36: (227, 119, 194), # bathtub
        39: (82, 84, 163),   # otherfurniture
        0: (0, 0, 0),        # unlabel/unknown
        255: (0, 0, 0),      # ignore label often maps to black
    }

def sparse_coords_to_dense_rgb_grid_nodevice(
    coords: torch.Tensor,            # Tensor coordinates (N, 3), integer type (long)
    labels: torch.Tensor,            # Tensor labels (N,), integer type (long)
    color_map: dict,                 # Dictionary mapping label IDs to RGB tuples (0-255)
    output_grid_size: tuple = None,  # Optional: Force specific (H, W, D). If None, calculated dynamically.
    coord_order: tuple = (1, 2, 0),  # Indices of (Z, Y, X) in coords columns. E.g., (0,1,2)=>(Z,Y,X)
    background_color: tuple = (0, 0, 0) # RGB tuple for empty voxels
    ) -> torch.Tensor:
    """
    Converts sparse voxel coordinates and labels to a dense RGB voxel grid.
    *** Assumes all input tensors are on the same device. No explicit device handling. ***

    Args:
        coords: Tensor (N, 3) of integer voxel indices. Must be non-negative.
        labels: Tensor (N,) of integer labels corresponding to coords.
        color_map: Dictionary mapping label IDs to RGB tuples (0-255).
        output_grid_size: Optional tuple (Height, Width, Depth). If None, grid size
                          is determined by the maximum coordinates in `coords`.
        coord_order: Tuple indicating which column in `coords` corresponds to (Z, Y, X).
                     Example: (0, 1, 2) means coords[:, 0] is Z, coords[:, 1] is Y, coords[:, 2] is X.
        background_color: RGB tuple (0-255) for background/empty voxels.

    Returns:
        torch.Tensor: Dense RGB voxel grid of shape (Height, Width, Depth, 3)
                      and dtype torch.uint8. Operations occur on the device of input tensors.
    """
    # --- Input Assertions and Type Checks ---
    assert coords.dim() == 2 and coords.shape[1] == 3, "Coords must be (N, 3)"
    assert labels.dim() == 1 and labels.shape[0] == coords.shape[0], "Labels must be (N,) and match coords count"
    assert len(coord_order) == 3 and all(i in [0, 1, 2] for i in coord_order), "coord_order must be permutation of (0, 1, 2)"
    assert len(set(coord_order)) == 3, "coord_order must have unique indices"
    if coords.dtype != torch.long:
        coords = coords.long()
    if labels.dtype != torch.long:
        labels = labels.long()

    num_voxels = coords.shape[0]

    # --- Handle Empty Input ---
    if num_voxels == 0:
        print("Warning: Input coords are empty.")
        if output_grid_size:
            h, w, d = output_grid_size
        else:
            h, w, d = 1, 1, 1
        # Create tensor on the same device as coords (implicitly)
        # Note: We need *some* tensor to infer the device if coords is empty.
        # If labels is also empty, this might default to CPU.
        # A truly robust version might need a device hint if input can be genuinely empty.
        ref_tensor = coords if coords.numel() > 0 else labels if labels.numel() > 0 else torch.tensor([])
        return torch.full((h, w, d, 3), background_color, dtype=torch.uint8, device=ref_tensor.device) # Infer device if possible

    # --- 1. Map Labels to Colors ---
    sparse_colors_list = [color_map.get(label.item(), background_color) for label in labels]
    sparse_colors = torch.tensor(sparse_colors_list, dtype=torch.uint8) # Let PyTorch place it based on default/context

    # --- 2. Determine Grid Size ---
    z_col, y_col, x_col = coord_order
    if output_grid_size:
        grid_h, grid_w, grid_d = output_grid_size
    else:
        # Operations like .max() happen on the tensor's device
        grid_h = torch.max(coords[:, y_col]).item() + 1
        grid_w = torch.max(coords[:, x_col]).item() + 1
        grid_d = torch.max(coords[:, z_col]).item() + 1

    # --- 3. Filter coordinates ---
    valid_mask = (coords[:, y_col] >= 0) & (coords[:, y_col] < grid_h) & \
                 (coords[:, x_col] >= 0) & (coords[:, x_col] < grid_w) & \
                 (coords[:, z_col] >= 0) & (coords[:, z_col] < grid_d)

    num_original = coords.shape[0]
    if not torch.all(valid_mask):
        coords = coords[valid_mask]
        sparse_colors = sparse_colors[valid_mask]
        num_valid = coords.shape[0]
        # print(f"Warning: Filtered {num_original - num_valid} coordinates outside grid bounds. {num_valid} remain.")
        if num_valid == 0:
            print("Warning: No valid coordinates remain after filtering.")
            ref_tensor = coords if coords.numel() > 0 else labels if labels.numel() > 0 else torch.tensor([])
            return torch.full((grid_h, grid_w, grid_d, 3), background_color, dtype=torch.uint8, device=ref_tensor.device)

    # --- 4. Create the Dense Grid Shell ---
    dense_rgb_grid = torch.zeros((grid_h, grid_w, grid_d, 3), dtype=torch.uint8)


    # --- 5. Place Sparse Colors ---
    h_indices = coords[:, y_col]
    w_indices = coords[:, x_col]
    d_indices = coords[:, z_col]

    # Indexing operation happens on the device where dense_rgb_grid, indices, and sparse_colors reside
    dense_rgb_grid[h_indices, w_indices, d_indices] = sparse_colors
    return dense_rgb_grid

def convert_voxel_to_2d_scan_image(
    dense_label_grid: torch.Tensor, # Input is now the dense grid of labels
    scale_factor: int = 2
):
    """
    Convert a 3D dense label grid to a 2D scan image by arranging patches
    colored by the dominant label of each Z-slice.

    Args:
    - dense_label_grid (torch.Tensor): Input 3D dense label grid (D, H, W), dtype long or int.
    - scale_factor (int): Scaling factor for resizing the colored patches.

    Returns:
    - torch.Tensor: 2D scan image (ImgH, ImgW, 3), dtype torch.float32 (range 0-1).
    """
    if dense_label_grid.numel() == 0 or dense_label_grid.shape[0] == 0:
        print("Warning: Input dense_label_grid is empty.")
        h = w = 14 * 8 * 8 # Default size from original code
        return torch.zeros((h, w, 3), dtype=torch.float32)

    # Get color map
    color_map = get_scannet_color_map()

    # --- Determine layout based on original logic ---
    image_h = image_w = 14 * 8 * 8
    grid_cols = max(1, 8 // scale_factor) # Number of patches per row
    patch_size = 14 * 8 * scale_factor # Size of patch area (including borders)
    # --- Assumption: Size of the content area within the patch ---
    content_size = 100 * scale_factor # Based on original code's `repeat` and placement calculation
    #-------------------------------------------------------------

    # Create an empty image tensor (using float for color values 0-1)
    image = torch.zeros((image_h, image_w, 3), dtype=torch.float32)

    num_slices = dense_label_grid.shape[0]
    loop_limit = int(16 * 4 / (scale_factor ** 2)) # Original loop limit

    slices_to_process = min(num_slices, loop_limit)
    if num_slices > loop_limit:
        print(f"Warning: Processing only the first {loop_limit} of {num_slices} slices due to original limit logic.")

    for i in range(slices_to_process):
        # Calculate top-left corner for the patch
        row_idx_in_grid = i // grid_cols
        col_idx_in_grid = i % grid_cols
        start_row = row_idx_in_grid * patch_size
        start_col = col_idx_in_grid * patch_size

        # Check bounds
        if start_row + patch_size > image_h or start_col + patch_size > image_w:
             print(f"Warning: Patch for slice {i} at ({start_row},{start_col}) exceeds image bounds. Skipping.")
             continue

        border_value = 1.0  # White border (float)
        border_size = 1

        # Draw borders for the patch area
        patch_border_area = image[start_row : start_row + patch_size, start_col : start_col + patch_size]
        patch_border_area[:border_size, :] = border_value  # Top
        patch_border_area[-border_size:, :] = border_value # Bottom
        patch_border_area[:, :border_size] = border_value  # Left
        patch_border_area[:, -border_size:] = border_value # Right

        # Get the current slice of labels
        label_slice = dense_label_grid[i] # Shape (H, W)

        # Find the most common label in this slice (ignoring label 0 if desired, but let's include it for now)
        unique_labels, counts = torch.unique(label_slice, return_counts=True)
        if len(counts) == 0: # Handle empty slices if they occur
            dominant_label_item = 0 # Default to 0 if slice is empty/all default
        else:
             dominant_label = unique_labels[torch.argmax(counts)]
             dominant_label_item = dominant_label.item()

        # Get color for the dominant label, normalize to 0-1
        label_color_rgb = color_map.get(dominant_label_item, (0, 0, 0)) # Get RGB tuple
        label_color_float = torch.tensor(label_color_rgb, dtype=torch.float32) / 255.0

        # Create a uniformly colored patch based on the *assumed* content size
        # Shape (content_size, content_size, 3)
        colored_patch = label_color_float.view(1, 1, 3).repeat(content_size, content_size, 1)

        # Calculate where to paste the content patch inside the bordered area
        paste_start_row = start_row + border_size
        paste_start_col = start_col + border_size
        paste_end_row = paste_start_row + content_size
        paste_end_col = paste_start_col + content_size

        # Ensure pasting coordinates are within the allocated patch area
        if paste_end_row <= start_row + patch_size and paste_end_col <= start_col + patch_size:
            image[paste_start_row : paste_end_row, paste_start_col : paste_end_col] = colored_patch
        else:
            # This case should ideally not happen if content_size <= patch_size - 2*border_size
            print(f"Warning: Calculated content placement for slice {i} exceeds patch bounds. Clipping.")
            # Calculate valid paste dimensions
            valid_h = max(0, min(content_size, start_row + patch_size - paste_start_row - border_size))
            valid_w = max(0, min(content_size, start_col + patch_size - paste_start_col - border_size))
            if valid_h > 0 and valid_w > 0:
                 image[paste_start_row : paste_start_row + valid_h, paste_start_col : paste_start_col + valid_w] = colored_patch[:valid_h, :valid_w, :]

    return image
def save_frame(image, filename):
    """Save an image to a file for debugging"""
    # Use PIL to save the image directly
    Image.fromarray(image).save(filename)
    print(f"Saved frame to {filename}")

def voxel_to_video(voxel_grid, output_path, fps=2, scale_factor=2):
    """Convert a voxel grid to a video where each frame is a vertical slice using MoviePy"""
    height, width, depth, channels = voxel_grid.shape
    print(f"Voxel grid shape: {voxel_grid.shape}")
    
    # Create a directory for debug frames
    os.makedirs("debug_frames", exist_ok=True)
    
    # We need 16 frames for an 8-second video at 2fps
    # Select 16 evenly spaced height levels
    frame_indices = np.linspace(0, height-1, 16, dtype=int)
    
    # Convert voxel grid to numpy
    voxel_np = voxel_grid.numpy()
    
    # Prepare all frames first
    frame_files = []
    for i, h_idx in enumerate(frame_indices):
        # Get the horizontal slice at this height (this is one "vertical block")
        slice_2d = voxel_np[h_idx, :, :, :]
        
        # Ensure slice is in uint8 format
        slice_2d_uint8 = slice_2d.astype(np.uint8)
       
        
        # Save frame for debugging and for video creation
        frame_path = f"debug_frames/frame_{i:02d}.png"
        save_frame(slice_2d_uint8, frame_path)
        frame_files.append(frame_path)

def visualize_2d_scan(image, filename='2d_scan_image.png'):
    """
    Visualize the 2D scan image using matplotlib. Saves the image.

    Args:
    - image (torch.Tensor): 2D scan image (H, W, C), float 0-1 or uint8 0-255.
    - filename (str): Path to save the visualization.
    """
    plt.figure(figsize=(12, 12)) # Adjusted size slightly
    # Ensure image is on CPU and is numpy array
    image_np = image.cpu().numpy()
    # Matplotlib handles float [0,1] or uint8 [0,255] automatically
    plt.imshow(image_np)
    plt.title('2D Scan Image (Color-coded by Slice Dominant Label)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved 2D scan image to {filename}")
    plt.close() # Close the plot to free memory


def main():
    # --- Configuration ---
    file_path = '/mnt/nas/bachvd/voxel-representation/data/scannet_3d/train/scene0000_00_vh_clean_2.pth'
    VOXEL_SIZE = 0.05 # Use uppercase for constants
    SCALE_FACTOR = 2 # For the 2D scan image generation
    OUTPUT_FILENAME = 'scan_image_scene0000_00.png'
    # ---------------------

    # Load point cloud data
    try:
        locs_in, feats_in, labels_in = torch.load(file_path, weights_only=False)
        print(f"Loaded data: locs={locs_in.shape}, feats={feats_in.shape}, labels={labels_in.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # --- Voxelization (Sparse Representation) ---
    voxelizer = Voxelizer(
        voxel_size=VOXEL_SIZE,
        clip_bound=None, # Or set appropriate bounds if needed
        use_augmentation=False
    )

    print("Voxelizing point cloud...")
    # return_ind=False by default in the provided Voxelizer snippet
    # Assuming voxelize returns coords, feats, labels, inds_reconstruct
    try:
        locs_voxel, feats_voxel, labels_voxel, _ = voxelizer.voxelize(
            locs_in,
            feats_in,
            labels_in
        )
    except Exception as e:
         print(f"Error during voxelization: {e}")
         # Potentially check Voxelizer implementation or input data types/ranges
         return

    print(f"Voxelization complete: {locs_voxel.shape[0]} unique voxels found.")

    if locs_voxel.shape[0] == 0:
        print("Error: No voxels generated. Cannot proceed.")
        return


    # --- Convert to Dense Grid ---
    # Determine grid size dynamically from sparse voxel coordinates
    # Add 1 because coordinates are 0-based indices (max_coord is the last index)
    grid_size_dense = (
        np.max(locs_voxel[:, 0]).item() + 1,
        np.max(locs_voxel[:, 1]).item() + 1,
        np.max(locs_voxel[:, 2]).item() + 1
    )
    print(f"Determined dense grid size: {grid_size_dense}")

    print("Creating dense label grid...")
    color_map = get_scannet_color_map()
    # Create dense grid using VOXELIZED coordinates and labels
    voxel_grid_labels = sparse_coords_to_dense_rgb_grid_nodevice(
        coords=torch.from_numpy(locs_voxel),
        labels=torch.from_numpy(labels_voxel),
        color_map=color_map,
        output_grid_size=None,
        # coord_order=(2,1,0),
        background_color=(25, 25, 25)
    )
    print(f"Dense label grid created: shape={voxel_grid_labels.shape}, dtype={voxel_grid_labels.dtype}")


    # --- Generate 2D Scan Image from Dense Grid ---
    print("Generating 2D scan image...")
    # scan_image = convert_voxel_to_2d_scan_image(
    #     dense_label_grid=voxel_grid_labels, # Pass the dense label grid
    #     scale_factor=SCALE_FACTOR
    # )
    voxel_to_video(voxel_grid_labels, output_path="debug.mp4")
    # print(f"2D scan image generated: shape={scan_image.shape}, dtype={scan_image.dtype}")

    # --- Visualize ---
    # visualize_2d_scan(scan_image, filename=OUTPUT_FILENAME)

if __name__ == '__main__':
    main()