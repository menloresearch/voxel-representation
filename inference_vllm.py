from dotenv import load_dotenv
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import base64
import os
from tqdm.auto import tqdm
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run voxel-based object detection with LLM")
    parser.add_argument("--checkpoint-path", required=True, 
                        help="Full path to the model checkpoint")
    parser.add_argument("--output-filename", required=True,
                        help="Path to save the output jsonl file")
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing 2d input images ex. output/test_100/image2d")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of images to process (process all by default)")
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv()  # take environment variables from .env
    
    # Initialize guided decoding parameters for JSON output
    guided_decoding_params = GuidedDecodingParams(json_object=True)
    
    # Load the model
    print(f"Loading model from: {args.checkpoint_path}")
    model = LLM(model=args.checkpoint_path)
    sampling_params = SamplingParams(
        guided_decoding=guided_decoding_params,
        max_tokens=2048, 
        temperature=0.0
    )
    
    # Process images
    input_dir = args.input_dir
    input_files = sorted(os.listdir(input_dir))
    
    if args.limit is not None:
        input_files = input_files[:args.limit]
        
    total = len(input_files)
    print(f"Processing {total} images from {input_dir}")
    
    messages_all = []
    for i in tqdm(range(len(input_files)), total=total):
        filename = f'{i}.png'
        image_path = os.path.join(input_dir, filename)
        
        # Convert image to base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    {"type": "text", "text": "Based on 2D Voxel scan, please predict objects in the 3D space in json format\n<start_of_image>"}
                ]
            }
        ]
        messages_all.append(messages)
    
    # Get model outputs
    outputs = model.chat(messages_all, sampling_params=sampling_params)
    
    # Process and save results
    error_count = 0
    with open(args.output_filename, 'w') as f:
        for example_id, output in enumerate(outputs):
            try:
                obj = json.loads(output.outputs[0].text)
                labels = {'id': example_id, 'objects': obj}
            except Exception as e:
                print(f'Error processing example {example_id}: {e}')
                error_count += 1
                labels = {'id': example_id, 'objects': []}
            
            f.write(json.dumps(labels) + '\n')
    
    print(f"Processing complete. {error_count} errors encountered.")
    print(f"Results saved to {args.output_filename}")


if __name__ == "__main__":
    main()