
from dataclasses import dataclass, field
from typing import List

from src.constants import Split


@dataclass
class DataGenerationConfig:
    max_obj_num: int = 10
    modelnet_path: str = ''
    split: str = 'train'
    skip_3d_image_gen: bool = True
    skip_3d_html_gen: bool = True
    skip_voxel_torch_save: bool = True
    max_height = 0.38
    max_length = 1.0
    voxel_size = 0.03 
    save_path: str = ''
    num_examples: int = 100
    scale_factors: List[int] = field(default_factory=list)


