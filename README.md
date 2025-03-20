## Voxel Representation Model

### Installation
```
conda create --prefix $(pwd)/.conda python=3.10
conda activate $(pwd)/.conda
pip install torch torchvision torchaudio
pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3
pip install accelerate python-dotenv matplotlib

pip install trimesh scipy omegaconf
```

### Run
```
export HF_HOME=$(pwd)/.hf_home
CUDA_VISIBLE_DEVICES=1 python main.py
```