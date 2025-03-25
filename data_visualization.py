import argparse
import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict

from src.constants import SAVE_VOXEL_NAME, VOXEL_FOLDER_NAME

def load_labels(labels_path: str, example_id: int) -> Dict:
    """Load a specific example from the labels.jsonl file."""
    with open(labels_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['id'] == example_id:
                return data
    raise ValueError(f"Example ID {example_id} not found in labels file")

def load_voxel_grid(data_path: str, example_id: int) -> torch.Tensor:
    """Load the voxel grid for the specified example"""
    try:
        voxel_grid = torch.load(os.path.join(data_path, VOXEL_FOLDER_NAME, SAVE_VOXEL_NAME.format(example_id=example_id)))
        
        return voxel_grid
    except FileNotFoundError:
        raise FileNotFoundError(f"file for example {example_id} not found")


def visualize_with_labels(voxel_grid: torch.Tensor, labels: Dict, output_path: str):
    """Visualize the voxel grid with object labels placed at the top,
    with lines down to voxel centers (marked by a red dot and coordinates)."""

    # Identify non-empty voxels and create occupancy grid
    rgb_sum = voxel_grid.sum(dim=-1)
    occupancy = (rgb_sum > 0).numpy().transpose(1, 2, 0)

    # Convert to numpy float in [0,1]
    colors = voxel_grid.numpy().astype(float) / 255.0
    colors = colors.transpose(1, 2, 0, 3)  # shape: (X, Y, Z, 3)

    # Create a (4)-channel RGBA array so we can control alpha independently
    colored_voxels = np.zeros(occupancy.shape + (4,), dtype=np.float32)
    for i in range(occupancy.shape[0]):
        for j in range(occupancy.shape[1]):
            for k in range(occupancy.shape[2]):
                if occupancy[i, j, k]:
                    # Copy over RGB
                    colored_voxels[i, j, k, :3] = colors[i, j, k]
                    # Set alpha to something less than 1 so we can see behind
                    colored_voxels[i, j, k, 3]  = 0.3

    # Create a matplotlib figure and a 3D axis
    fig = plt.figure(figsize=(12, 10))
    ax: Any = fig.add_subplot(111, projection='3d')

    # Plot the voxels with transparency and no edges
    ax.voxels(
        occupancy,
        facecolors=colored_voxels,
        edgecolor=None  # Removes the black gridlines so things behind show better
    )

    # Set axis limits and title
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 64)
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis (height)")
    ax.set_title(f"Voxel Visualization for Example {labels['id']}")

    zmin, zmax = ax.get_zlim()
    for obj in labels.get('objects', []):
        if 'voxel_coords_center' in obj:
            cx = obj['voxel_coords_center']['x']
            cy = obj['voxel_coords_center']['y']
            cz = obj['voxel_coords_center']['z']
            
            z_label = zmax + 3
            
            # Label text above
            ax.text(
                cx, cy, z_label,
                f"{obj['description']} ({obj['color']})",
                color='black',
                fontsize=9,
                weight='bold',
                ha='center',
                va='bottom',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5')
            )
            
            # Gray line from label down to center
            ax.plot([cx, cx], [cy, cy], [z_label, cz], color='gray', linestyle='-')
            
            # Red dot at center, no depth shading + higher zorder
            ax.scatter(
                [cx], [cy], [cz],
                color='red', s=40,
                depthshade=False,
                zorder=10
            )
            
            # Coordinate text, slightly above and with higher zorder
            ax.text(
                cx, cy, cz,
                f"({cx}, {cy}, {cz})",
                color='red',
                fontsize=8,
                weight='bold',
                ha='center',
                va='top',
                zorder=11,
                bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2')
            )

    # Add legend-like text on the 2D plane
    legend_text = []
    for i, obj in enumerate(labels.get('objects', [])):
        legend_text.append(f"{i+1}: {obj['description']} - {obj['color']}")
    if legend_text:
        ax.text2D(
            0.05, 0.95,
            "\n".join(legend_text),
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    ax.view_init(elev=30, azim=-45)  # Optional viewpoint

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize voxel grid with labels")
    parser.add_argument("--data-path", type=str, required=True, help="Path to data directory")
    parser.add_argument("--example-id", type=int, required=True, help="Example ID to visualize")
    parser.add_argument("--labels-file", type=str, help="Path to custom labels file (default: labels.jsonl in data directory)")
    args = parser.parse_args()
    
    # Determine the labels path
    if args.labels_file:
        # Use the custom labels file path provided
        labels_path = args.labels_file
    else:
        # Default to labels.jsonl in the data directory
        labels_path = os.path.join(args.data_path, "labels.jsonl")
    
    try:
        # Load the labels for the specified example
        labels = load_labels(labels_path, args.example_id)
        
        # Load or reconstruct the voxel grid for the specified example
        voxel_grid = load_voxel_grid(args.data_path, args.example_id)
        if args.labels_file:
            
            label_name = os.path.splitext(os.path.basename(args.labels_file))[0]
            output_path = os.path.join(args.data_path, f"visualized_{label_name}_{args.example_id}.png")
        else:
            output_path = os.path.join(args.data_path, f"visualized_labels_{args.example_id}.png")

        visualize_with_labels(voxel_grid, labels, output_path)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())