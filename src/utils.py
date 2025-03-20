from typing import Any
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import os

from src.config import DataGenerationConfig
from src.constants import IMAGE_2D_FOLDER_NAME, IMAGE_3D_FOLDER_NAME, SAVE_3D_IMAGE_NAME, SAVE_2D_IMAGE_NAME
from PIL import Image

def init_voxel():
    # Create an empty voxel grid with RGB channels (height, width, depth, 3)
    # The 3 channels are: R, G, B using 0-255 range
    voxel_grid = torch.zeros((16, 100, 100, 3), dtype=torch.uint8)
    return voxel_grid

def convert_voxel_to_2d_scan_image(voxel_grid: torch.Tensor):
    image = torch.zeros((14 * 8 * 8, 14 * 8 * 8, 3), dtype=torch.long)
    SCALE_FACTOR = 2
    for i in range(len(voxel_grid)):
        x = (i // (8 // SCALE_FACTOR)) * 14 * 8 * SCALE_FACTOR
        y = (i % (8 // SCALE_FACTOR)) * 14 * 8 * SCALE_FACTOR
        
        border_value = 255
        border_size = 1
        # patch = image[x+border_size:x+100+border_size, y+border_size:y+100+1]
        patch_border = image[x:x+14 * 8 * SCALE_FACTOR, y:y+14 * 8 * SCALE_FACTOR]
        patch_border[:border_size, :] = border_value  # Top border
        patch_border[-border_size:, :] = border_value  # Bottom border
        patch_border[:, :border_size] = border_value  # Left border
        patch_border[:, -border_size:] = border_value  # Right border

        voxel_grid_2d = voxel_grid[i]
        voxel_grid_2d_for_resize = voxel_grid_2d.permute(2, 0, 1).unsqueeze(0)  # Shape becomes [1, 3, 100, 100]

        # Resize the voxel_grid_2d
        resized_voxel_grid_2d = F.interpolate(voxel_grid_2d_for_resize, scale_factor=SCALE_FACTOR, mode='bilinear', align_corners=False)

        # Convert back to the original format
        resized_voxel_grid_2d = resized_voxel_grid_2d.squeeze(0).permute(1, 2, 0)  # Shape becomes [200, 200, 3]
        image[x+border_size:x+100 * SCALE_FACTOR+ border_size, y + border_size:y+100 * SCALE_FACTOR+ border_size] = resized_voxel_grid_2d
    return image



def visualize_voxel_grid(voxel_grid: torch.Tensor, example_id: int, cfg: DataGenerationConfig):
    # Identify non-empty voxels and create occupancy grid
    # A voxel is occupied if any RGB channel has a value
    rgb_sum = voxel_grid.sum(dim=-1)
    occupancy = (rgb_sum > 0).numpy().transpose(1, 2, 0)
    
    # Normalize colors to 0-1 range for matplotlib
    colors = voxel_grid.numpy().astype(float) / 255.0
    colors = colors.transpose(1, 2, 0, 3)
    
    # Create a color array for the voxels
    colored_voxels = np.zeros(occupancy.shape + (3,), dtype=np.float32)
    
    # Assign colors only to occupied voxels
    for i in range(occupancy.shape[0]):
        for j in range(occupancy.shape[1]):
            for k in range(occupancy.shape[2]):
                if occupancy[i, j, k]:
                    colored_voxels[i, j, k] = colors[i, j, k]

    # Create a matplotlib figure and a 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax: Any = fig.add_subplot(111, projection='3d')

    # Use matplotlib's voxel plotting function with facecolors
    ax.voxels(occupancy, facecolors=colored_voxels, edgecolor='k', alpha=0.9)

    # Set axis limits for better visualization
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 64)

    # Label the axes and set a title
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis (height)")
    ax.set_title("Colored Voxel Chair Visualization")

    # Display the plot
    save_path_img = os.path.join(cfg.save_path, IMAGE_3D_FOLDER_NAME, SAVE_3D_IMAGE_NAME.format(example_id=example_id))
    plt.savefig(save_path_img)

def save_2d_image(voxel_grid: torch.Tensor, example_id: int, cfg: DataGenerationConfig):
    image = convert_voxel_to_2d_scan_image(voxel_grid)

    # Convert to NumPy and ensure it's in the correct format
    image_np = image.numpy()  # Convert to NumPy
    save_path_img = os.path.join(cfg.save_path, IMAGE_2D_FOLDER_NAME, SAVE_2D_IMAGE_NAME.format(example_id=example_id))
    Image.fromarray(np.uint8(image_np)).save(save_path_img)