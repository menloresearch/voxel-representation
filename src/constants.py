from typing import Literal


COLOR_MAP = [
    {"name": "red", "rgb": [255, 0, 0]},
    {"name": "green", "rgb": [0, 255, 0]},
    {"name": "blue", "rgb": [0, 0, 255]},
    {"name": "yellow", "rgb": [255, 255, 0]},
    {"name": "magenta", "rgb": [255, 0, 255]},
    {"name": "cyan", "rgb": [0, 255, 255]},
    {"name": "maroon", "rgb": [128, 0, 0]},
    {"name": "dark_green", "rgb": [0, 128, 0]},
    {"name": "navy", "rgb": [0, 0, 128]},
    {"name": "olive", "rgb": [128, 128, 0]},
    {"name": "purple", "rgb": [128, 0, 128]},
    {"name": "teal", "rgb": [0, 128, 128]},
    {"name": "gray", "rgb": [128, 128, 128]},
    {"name": "orange", "rgb": [255, 165, 0]},
    {"name": "chocolate", "rgb": [210, 105, 30]},
]

SELECTED_OBJ_CATEGORIES = [
    'toilet',
    'airplane',
    'bathtub',
    'bottle',
    'bowl',
    'cone',
    'cup',
    'desk',
    'guitar',
    'laptop',
    'plant',
    'sofa',
    'stool',
    'tent',
    'toilet'
]

Split = Literal['train', 'test']

IMAGE_2D_FOLDER_NAME = 'image2d'
IMAGE_3D_FOLDER_NAME = 'image3d'
IMAGE_3D_HTML_FOLDER_NAME = 'html_obj_3d'

SAVE_2D_IMAGE_NAME = '{example_id}.png'
SAVE_3D_IMAGE_NAME = '{example_id}.png'
SAVE_3D_HTML_NAME = '{example_id}.html'