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


## Getting Started

### Run Inference for 100 examples

```
CUDA_VISIBLE_DEVICES=0 python inference_vllm.py --checkpoint-path homebrewltd/voxel-representation-gemma3-4b --input-dir output/test_100/image2d --output-filename output/test_100/model_prediction.jsonl
```
### Visualize

#### Labels

```
python data_visualization.py --data-path output/test_100 --example-id  1
```

#### Model Prediction

```
python data_visualization.py --data-path output/test_100 --example-id  1 --labels-file output/test_100/model_prediction.jsonl
```

## Getting Eval Result

### Run Inference for test set

```
CUDA_VISIBLE_DEVICES=0 python inference_vllm.py --checkpoint-path homebrewltd/voxel-representation-gemma3-4b --input-dir output/test/image2d --output-filename output/test/model_prediction.jsonl
```

### Run Eval

```
python data_eval.py --label output/test/labels.jsonl  --output output/test/model_prediction.jsonl
```