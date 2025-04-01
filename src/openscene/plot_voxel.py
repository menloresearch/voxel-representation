import torch
import numpy as np
import plotly.graph_objs as go
from src.openscene.voxelizer import Voxelizer


def get_scannet_color_map():
    """
    Define a color map for ScanNet dataset object classes.
    These are typical colors for common indoor scene objects.
    """
    return {
        1: (174., 199., 232.),
        2: (152., 223., 138.),
        3: (31., 119., 180.),
        4: (255., 187., 120.),
        5: (188., 189., 34.),
        6: (140., 86., 75.),
        7: (255., 152., 150.),
        8: (214., 39., 40.),
        9: (197., 176., 213.),
        10: (148., 103., 189.),
        11: (196., 156., 148.),
        12: (23., 190., 207.),
        14: (247., 182., 210.),
        16: (219., 219., 141.),
        24: (255., 127., 14.),
        28: (158., 218., 229.),
        33: (44., 160., 44.),
        34: (112., 128., 144.),
        36: (227., 119., 194.),
        39: (82., 84., 163.),
        0: (0., 0., 0.), # unlabel/unknown
        255: (0.,0.,0.),
    }
def rgb_to_hex(rgb):
    """
    Convert RGB tuple to hex color code.
    
    Args:
    - rgb (tuple): RGB values (0-255 range)
    
    Returns:
    - str: Hex color code
    """
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
def rgb_to_normalized(rgb):
    """
    Convert RGB tuple to normalized values (0-1 range)
    
    Args:
    - rgb (tuple): RGB values (0-255 range)
    
    Returns:
    - tuple: Normalized RGB values
    """
    return tuple(x/255.0 for x in rgb)

def load_and_plot_point_cloud(file_path, voxel_size=0.05, plot_original=True, plot_voxelized=True):
    """
    Load point cloud data, voxelize it, and create an interactive 3D plot.
    
    Args:
    - file_path (str): Path to the .pth file containing point cloud data
    - voxel_size (float): Size of voxels for quantization
    - plot_original (bool): Whether to plot the original point cloud
    - plot_voxelized (bool): Whether to plot the voxelized point cloud
    
    Returns:
    - Plotly figure object
    """
    # Load the point cloud data
    locs_in, feats_in, labels_in = torch.load(file_path, weights_only=False)
    labels_in[labels_in == -100] = 255
    labels_in = labels_in.astype(np.uint8)
    feats_in = (feats_in + 1.) * 127.5
    
    # Get color map
    color_map = get_scannet_color_map()
    
    # Create a voxelizer
    voxelizer = Voxelizer(
        voxel_size=voxel_size,
        clip_bound=None,
        use_augmentation=False
    )
    
    # Voxelize the point cloud
    locs_voxel, feats_voxel, labels_voxel, _ = voxelizer.voxelize(
        locs_in, 
        feats_in, 
        labels_in
    )
    
    fig = go.Figure()
    
    # Plot original point cloud
    if plot_original:
        point_colors = [f'rgb({int(color_map.get(label, (0,0,0))[0])},{int(color_map.get(label, (0,0,0))[1])},{int(color_map.get(label, (0,0,0))[2])})' for label in labels_in]
        scatter_original = go.Scatter3d(
            x=locs_in[:, 0], 
            y=locs_in[:, 1], 
            z=locs_in[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                color=point_colors,
                opacity=0.6
            ),
            name='Original Point Cloud'
        )
        fig.add_trace(scatter_original)
    
    # Plot voxelized point cloud
    if plot_voxelized:
        voxel_point_colors = [f'rgb({int(color_map.get(label, (0,0,0))[0])},{int(color_map.get(label, (0,0,0))[1])},{int(color_map.get(label, (0,0,0))[2])})' for label in labels_voxel]
        scatter_voxel = go.Scatter3d(
            x=locs_voxel[:, 0], 
            y=locs_voxel[:, 1], 
            z=locs_voxel[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=voxel_point_colors,
                opacity=0.8
            ),
            name='Voxelized Point Cloud'
        )
        fig.add_trace(scatter_voxel)
    
    # Update layout
    fig.update_layout(
        title='3D Point Cloud Visualization',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        height=800,
        width=1200
    )
    
    return fig


def create_class_legend_html(color_map):
    """
    Create an HTML legend with class colors and names.
    
    Args:
    - color_map (dict): Dictionary of class IDs to RGB color tuples
    
    Returns:
    - str: HTML string for the legend
    """
    # Define class names mapping
    class_names = {
        1: 'Wall', 
        2: 'Floor', 
        3: 'Cabinet', 
        4: 'Bed', 
        5: 'Chair', 
        6: 'Sofa', 
        7: 'Table', 
        8: 'Door', 
        9: 'Window', 
        10: 'Bookshelf', 
        11: 'Picture', 
        12: 'Counter', 
        14: 'Desk', 
        16: 'Curtain', 
        24: 'Refrigerator', 
        28: 'Shower Curtain', 
        33: 'Toilet', 
        34: 'Sink', 
        36: 'Other Furniture', 
        39: 'Bathtub',
        0: 'Unlabeled',
        255: 'Ignore Label'
    }
    
    # Create legend HTML
    legend_html = "<div style='position:fixed;top:10px;right:10px;background:white;border:1px solid black;padding:10px;z-index:1000;'>"
    legend_html += "<h3>Object Class Colors</h3>"
    legend_html += "<ul style='list-style-type:none;padding:0;'>"
    
    # Sort class IDs for consistent display
    sorted_classes = sorted(color_map.keys())
    
    for class_id in sorted_classes:
        # Skip certain class IDs if they don't have a name
        if class_id not in class_names:
            continue
        
        # Convert RGB to hex color
        color = color_map[class_id]
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))
        
        # Get class name, fallback to generic name if not found
        class_name = class_names.get(class_id, f'Class {class_id}')
        
        # Add legend item
        legend_html += (
            f"<li style='margin-bottom:5px;'>"
            f"<span style='background-color:{hex_color};width:20px;height:20px;display:inline-block;margin-right:10px;border:1px solid black;'></span>"
            f"{class_name}"
            f"</li>"
        )
    
    legend_html += "</ul></div>"
    return legend_html

def main():
    file_path = '/mnt/nas/bachvd/voxel-representation/data/scannet_3d/train/scene0000_00_vh_clean_2.pth'
    
    # Create and show the plot
    fig = load_and_plot_point_cloud(
        file_path, 
        voxel_size=0.05,  # Adjust voxel size as needed
        plot_original=True, 
        plot_voxelized=True
    )
    
    # Add a color legend
    color_map = get_scannet_color_map()
    legend_html = create_class_legend_html(color_map)
    
    # Save the figure as an interactive HTML with legend
    fig.write_html("point_cloud_visualization.html", include_plotlyjs=True, full_html=True, 
                   auto_open=False, include_mathjax=False)
    
    with open("point_cloud_visualization.html", 'r') as f:
        html_content = f.read()
    
    with open("point_cloud_visualization.html", 'w') as f:
        html_content = html_content.replace('</body>', f'{legend_html}</body>')
        f.write(html_content)
main()