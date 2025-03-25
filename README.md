# Voxel Representation Model

## Installation
```
conda create -y --prefix $(pwd)/.conda python=3.10
conda activate $(pwd)/.conda
pip install torch torchvision torchaudio
pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3
pip install accelerate python-dotenv matplotlib trimesh scipy omegaconf vllm
conda install -y -c conda-forge libstdcxx-ng
```

### For Training
```
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
git checkout 59a56f7226f24b3b8c37b6a4da0a5802b4022ead
pip install -e ".[torch,metrics]"
cd ..
pip install deepspeed liger-kernel
``` 

## Getting Started

### Run Inference for 100 examples

```
export HF_HOME=$(pwd)/.hf_home
CUDA_VISIBLE_DEVICES=0 python inference_vllm.py --checkpoint-path homebrewltd/voxel-representation-gemma3-4b --input-dir output/test_100/image2d --output-filename output/test_100/model_prediction.jsonl
```
### Visualize

#### Labels

```
python data_visualization.py --data-path output/test_100 --example-id <id>
```
- id can be 0-99

#### Model Prediction

```
python data_visualization.py --data-path output/test_100 --labels-file output/test_100/model_prediction.jsonl --example-id <id> 
```
- id can be 0-99

## Getting Eval Result

### Run Inference for test set

```
export HF_HOME=$(pwd)/.hf_home
CUDA_VISIBLE_DEVICES=0 python inference_vllm.py --checkpoint-path homebrewltd/voxel-representation-gemma3-4b --input-dir output/test/image2d --output-filename output/test/model_prediction.jsonl
```

### Run Eval

```
python data_eval.py --label output/test/labels.jsonl  --output output/test/model_prediction.jsonl
```

## Training

### Data Preparation

#### Download [ModelNet 40 Dataset](https://modelnet.cs.princeton.edu/)
```
wget http://modelnet.cs.princeton.edu/ModelNet40.zip
unzip ModelNet40.zip
```

#### Generate Synthesis data
```
# edit your path to model net in  config/data_train.yaml 
cp config/data_train.example.yaml config/data_train.yaml 
python data_generation.py
```

#### Convert Synthesis data to LLaMA-Factory format
```
python process_data.py output/train
```

### Training
```
export HF_HOME=$(pwd)/.hf_home
DISABLE_VERSION_CHECK=1 llamafactory-cli  train ./config/training_gemma3_pt_visual.yaml
```

### Push to hub
```
python push_to_hub.py <username> <checkpoint_number> saves/gemma3-4b-pt/full/sft/checkpoint-1100
```