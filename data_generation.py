import json
import torch
import numpy as np
import random

import trimesh
import os
import numpy as np
import glob
from trimesh.viewer import scene_to_html
from tqdm.auto import tqdm
from omegaconf import OmegaConf
import logging

from src.config import DataGenerationConfig
from src.constants import COLOR_MAP, IMAGE_2D_FOLDER_NAME, IMAGE_3D_FOLDER_NAME, IMAGE_3D_HTML_FOLDER_NAME, SAVE_2D_IMAGE_NAME, SAVE_3D_HTML_NAME, SAVE_3D_IMAGE_NAME, SELECTED_OBJ_CATEGORIES, VOXEL_FOLDER_NAME, Split
from src.utils import init_voxel, save_2d_image, save_voxel_grid, visualize_voxel_grid


def load_mesh(voxel_grid: torch.Tensor, split: Split, example_id: int, cfg: DataGenerationConfig):
    num_objects_to_use = random.randint(1, cfg.max_obj_num)
    
    # List to store selected files with their categories
    selected_files_with_categories = []
    
    # Select random files from random categories
    for _ in range(num_objects_to_use):
        # Randomly select a category
        selected_category = random.choice(SELECTED_OBJ_CATEGORIES)

        category_dir = os.path.join(cfg.modelnet_path, selected_category, split)
        category_files = glob.glob(os.path.join(category_dir, "*.off"))
        
        if category_files:
            selected_file = random.choice(category_files)
            selected_files_with_categories.append((selected_file, selected_category))
    
    print(f"Randomly selected {len(selected_files_with_categories)} objects from multiple categories")
    
    html3d_path = None if cfg.skip_3d_html_gen else os.path.join(IMAGE_3D_HTML_FOLDER_NAME, SAVE_3D_HTML_NAME.format(example_id=example_id))
    image3d_path = None if cfg.skip_3d_image_gen else os.path.join(IMAGE_3D_FOLDER_NAME, SAVE_3D_IMAGE_NAME.format(example_id=example_id))
    # Result dictionary
    result = {
        "id": example_id,
        "image2d_path": os.path.join(IMAGE_2D_FOLDER_NAME, SAVE_2D_IMAGE_NAME.format(example_id=example_id)),
        "image3d_path": image3d_path,
        "html3d_path": html3d_path,
        "number_of_objects": len(selected_files_with_categories),
        "objects": []
    }
    # Create a scene to hold all meshes
    scene = trimesh.Scene()
    mesh_list = []
    face_offset = 0
    object_face_ranges = []  # Will store (start_face, end_face, object_index)

    # Create a floor plane
    floor_size = 9.0  # Size of the floor (adjust as needed)
    floor_vertices = np.array([
        [-floor_size/2, -floor_size/2, 0.0],
        [floor_size/2, -floor_size/2, 0.0],
        [floor_size/2, floor_size/2, 0.0],
        [-floor_size/2, floor_size/2, 0.0]
    ])
    floor_faces = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ])
    floor = trimesh.Trimesh(vertices=floor_vertices, faces=floor_faces)
    
    # Set floor color (e.g., light gray)
    floor_color = [200, 200, 200]
    floor.visual.face_colors = np.tile(floor_color, (len(floor.faces), 1)) # type: ignore
    
    # Add floor to the scene and mesh list
    scene.add_geometry(floor, node_name="floor")

    # Load each mesh, normalize its size, and position it randomly
    placed_objects = []
    obj_count = 0
    for i, (file_path, category) in enumerate(selected_files_with_categories):
        try:
            mesh = trimesh.load_mesh(file_path)
            print(f"Loaded: {category} - {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
        else:
            # Assign color
            color_info = random.choice(COLOR_MAP)
            color_name = color_info["name"]
            color_rgb = color_info["rgb"]
            
            face_colors = np.tile(color_rgb, (len(mesh.faces), 1))
            mesh.visual.face_colors = face_colors # type: ignore

            current_x, current_y, current_height = mesh.bounding_box.extents
            
            # Calculate scale factors for each constraint
            height_scale = cfg.max_height / current_height if current_height > cfg.max_height else 1.0
            x_scale = cfg.max_length / current_x if current_x > cfg.max_length else 1.0
            y_scale = cfg.max_length / current_y if current_y > cfg.max_length else 1.0
    
            # Use the smallest scale factor to ensure all constraints are met
            scale_factor = min(height_scale, x_scale, y_scale)
                
            # current_size = mesh.bounding_box.extents[2]
            # scale_factor = max_h / current_size
            mesh.apply_scale(scale_factor)
            
            # Center the mesh on its centroid before transforming
            mesh.apply_translation(-mesh.centroid)
            
            # Apply random rotation
            angle = np.random.uniform(0, 2 * np.pi)  # ∂Random angle between 0 and 2π radians
            rotation_matrix = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
            mesh.apply_transform(rotation_matrix)

            # Apply random scaling variance
            scale_variance = np.random.uniform(0.5, 1.0)
            mesh.apply_scale(scale_variance)

            min_z = mesh.vertices[:, 2].min()
            mesh.apply_translation([0, 0, -min_z])
            
            bounds = mesh.bounding_box.bounds

            max_attempts = 50
            placed = False
            
            for attempt in range(max_attempts):
                # Generate random position
                random_position_x = np.random.uniform(0, 2.85 - mesh.bounding_box.extents[0], size=1)
                random_position_y = np.random.uniform(0, 2.85 - mesh.bounding_box.extents[1], size=1)
                random_position = np.concatenate((random_position_x, random_position_y, [0]))
                
                # Calculate new bounds after translation
                new_min_bound = bounds[0] + random_position
                new_max_bound = bounds[1] + random_position
                
                # Check for overlap with existing objects
                overlap = False
                for existing_bounds in placed_objects:
                    # Check if boxes overlap in all dimensions
                    if (new_min_bound[0] < existing_bounds[1][0] and 
                        new_max_bound[0] > existing_bounds[0][0] and
                        new_min_bound[1] < existing_bounds[1][1] and 
                        new_max_bound[1] > existing_bounds[0][1]):
                        overlap = True
                        break
                
                if not overlap:
                    # Position is valid, apply it
                    mesh.apply_translation(random_position)
                    placed_objects.append((new_min_bound, new_max_bound))
                    placed = True
                    
                    # In order to know which faces belong to this object, we record:
                    num_faces = len(mesh.faces)
                    object_face_ranges.append((face_offset, face_offset + num_faces, obj_count))
                    face_offset += num_faces
                    obj_count += 1
                    
                    break
            
            if not placed:
                print(f"Could not place object {i} after {max_attempts} attempts")
                continue
            
            mesh_list.append(mesh)
            scene.add_geometry(mesh, node_name=f"{category}_{i}")

            # Voxelize this individual mesh
            individual_voxels = mesh.voxelized(cfg.voxel_size)
            
            # Count occupied voxels and find center
            voxel_count = individual_voxels.matrix.sum()
            
            if voxel_count > 0:
                # Add object information to result
                object_info = {
                    'id': str(i),
                    "color": color_name,
                    "description": category,
                    'voxel_coords': [],
                    "number_of_occupied_voxel": int(voxel_count)
                }
                result["objects"].append(object_info)
    
    if not cfg.skip_3d_html_gen:
        html = scene_to_html(scene)
        save_path_3d_html = os.path.join(cfg.save_path, IMAGE_3D_HTML_FOLDER_NAME, SAVE_3D_HTML_NAME.format(example_id=example_id))
        with open(save_path_3d_html, 'w') as f:
            f.write(html)    

    # Combine all meshes into a single mesh
    if len(mesh_list) > 0:
        combined_mesh = trimesh.util.concatenate(mesh_list)
        print(f"Successfully combined {len(mesh_list)} meshes into one mesh with "
              f"{len(combined_mesh.vertices)} vertices and {len(combined_mesh.faces)} faces")

    else:
        raise ValueError("No meshes were successfully loaded")
    voxels = combined_mesh.voxelized(cfg.voxel_size)
    # Get voxel grid as a binary array
    voxel_points = voxels.points
    
    # Get the colors for each voxel from the nearest face
    nearest_faces = combined_mesh.nearest.on_surface(voxel_points)[2]
    voxel_colors = combined_mesh.visual.face_colors[nearest_faces][:, :3]  # Get RGB only

    object_assignments = np.full(len(nearest_faces), -1, dtype=int)
    for (start_face, end_face, obj_idx) in object_face_ranges:
        # face indices for this object are [start_face, end_face)
        in_range = (nearest_faces >= start_face) & (nearest_faces < end_face)
        object_assignments[in_range] = obj_idx
    
    # Convert points to voxel coordinates
    vox_coords = np.floor(voxel_points / cfg.voxel_size).astype(int)

    # Find the minimum coordinates in each dimension
    min_coords = np.min(vox_coords, axis=0)

    # Normalize to start from zero
    normalized_vox_coords = vox_coords - min_coords

    # Calculate dimensions of the voxelized model
    max_coords = np.max(normalized_vox_coords, axis=0)  # Get maximum coordinates after normalization
    model_dimensions = max_coords + 1  # Add 1 because coordinates are 0-indexed
    
    # Get dimensions
    vg_height, vg_width, vg_depth, _ = voxel_grid.shape
    
    # Calculate available space for random placement
    available_space = np.array([vg_width, vg_depth, vg_height]) - model_dimensions

    # Generate random offsets within available space (if there's room)
    if np.all(available_space > 0):
        # Create random offsets for each dimension
        w_offset = np.random.randint(0, available_space[0] + 1)
        d_offset = np.random.randint(0, available_space[1] + 1)
        h_offset = 0
        
        random_offsets = np.array([w_offset, d_offset, h_offset])
        
        # Apply offsets to normalized coordinates
        placed_vox_coords = normalized_vox_coords + random_offsets
    else:
        # If the model is too large to be randomly placed, center it
        placed_vox_coords = normalized_vox_coords + (available_space / 2).astype(int)

    # Fill voxel_grid with the assigned colors
    for i, coord in enumerate(placed_vox_coords):
        w, d, h = coord
        # Check bounds
        if 0 <= h < vg_height and 0 <= w < vg_width and 0 <= d < vg_depth:
            voxel_grid[h, w, d] = torch.tensor(voxel_colors[i])
    

    # store these voxel indices in each object’s dictionary
    # so we know exactly which voxel belongs to which object:
    for voxel_idx, obj_idx in enumerate(object_assignments):
        if obj_idx < 0:
            continue
        coord = placed_vox_coords[voxel_idx]
        
        # Append or store however you like: e.g. as a list of (h, w, d) tuples
        result["objects"][obj_idx]["voxel_coords"].append(tuple(coord))
    
     # Calculate center coordinates for each object
    for obj in result["objects"]:
        voxels = obj["voxel_coords"]
        # Calculate the mean of all voxel coordinates to get the center
        voxel_array = np.array(voxels)
        center = voxel_array.mean(axis=0)
        
        # Store center in the desired format
        obj["voxel_coords_center"] = {
            "x": int(center[0]),  # width
            "y": int(center[1]),  # depth
            "z": int(center[2])   # height
        }
    
    # Calculate bounding box for each object
    # for obj in result["objects"]:
    #     voxels = obj["voxel_coords"]
    #     # Convert to numpy array for calculations
    #     voxel_array = np.array(voxels)
        
    #     min_coords = np.min(voxel_array, axis=0)
    #     max_coords = np.max(voxel_array, axis=0)
        
    #     # Store bounding box in the desired format
    #     obj["voxel_bbox"] = {
    #         "x": int(min_coords[0]),  # width
    #         "y": int(min_coords[1]),  # depth
    #         "z": int(min_coords[2]),   # height
    #         "x_width": int(max_coords[0] - min_coords[0]),  # width
    #         "y_depth": int(max_coords[1] - min_coords[1]),  # depth
    #         "z_height": int(max_coords[2] - min_coords[2])   # height
    #     }
        
        # Remove the original voxel coordinates to save space
        del obj["voxel_coords"]
    
    print(f"""Filled voxel grid with matrix of shape {
        (
            int(model_dimensions[2]), 
            int(model_dimensions[0]), 
            int(model_dimensions[1])
        )} centered in grid of shape {voxel_grid.shape[:-1]}""")
    return result


def generate_one_example(split: Split, example_id: int, cfg: DataGenerationConfig):
    voxel_grid = init_voxel()
    result = load_mesh(voxel_grid, split, example_id, cfg)
    
    print(f"Example {example_id}: {result}")

    if not cfg.skip_3d_image_gen:
        visualize_voxel_grid(voxel_grid,  example_id, cfg)
    
    if not cfg.skip_voxel_torch_save:
        save_voxel_grid(voxel_grid, example_id, cfg)
    
    save_2d_image(voxel_grid, example_id, cfg)
    
    with open(os.path.join(cfg.save_path, 'labels.jsonl'), 'a') as f:
        f.write(json.dumps(result) + '\n')

def generate_examples(cfg: DataGenerationConfig, split: Split = 'train'):
    """Generate multiple examples with a progress bar"""

    os.makedirs(cfg.save_path, exist_ok=False)

    os.makedirs(os.path.join(cfg.save_path, IMAGE_3D_HTML_FOLDER_NAME), exist_ok=False)
    os.makedirs(os.path.join(cfg.save_path, IMAGE_3D_FOLDER_NAME), exist_ok=False)
    
    for scale_factor in cfg.scale_factors:
        os.makedirs(os.path.join(cfg.save_path, IMAGE_2D_FOLDER_NAME.format(scale_factor=scale_factor)), exist_ok=False)
    
    os.makedirs(os.path.join(cfg.save_path, VOXEL_FOLDER_NAME), exist_ok=False)
    
    for example_id in tqdm(range(cfg.num_examples), desc=f"Generating {split} examples"):
        generate_one_example(split, example_id, cfg)



def setup_config() -> DataGenerationConfig:
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(DataGenerationConfig())
    cfg_merged = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg_merged)  # type: ignore
    return cfg  # type: ignore

def main():
    # Configure logging
    logging.basicConfig(filename='error.log', level=logging.ERROR, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Load the configuration
    cfg = setup_config()
    
    # Generate multiple examples
    generate_examples(cfg, split=cfg.split) # type: ignore

if __name__ == '__main__':
    main()