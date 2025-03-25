import json
import os
import os
import sys
import shutil


def convert_jsonl_to_json(input_file, output_file, image_dir=""):
    """
    Converts JSONL file to the required JSON format, only including entries
    where the referenced 2D image actually exists.
    
    Args:
        input_file: Path to the labels.jsonl file
        output_file: Path to save the output data.json
        image_dir: Base directory for images (if different from the current directory)
    """
    output_data = []
    skipped_count = 0
    processed_count = 0
    
    # Read the JSONL file
    with open(input_file, 'r') as f:
        for line in f:
            # Parse each line as JSON
            item = json.loads(line.strip())
            
            # Check if the image exists
            image_path = os.path.join(image_dir, item["image2d_path"])
            if not os.path.exists(image_path):
                skipped_count += 1
                continue
            
            # Format the objects as a string
            objects_list = []
            for obj in item["objects"]:
                objects_list.append(obj)
            
            # Convert objects to a formatted string
            objects_str = json.dumps(objects_list, indent=2)
            
            # Create the output item with the specified format
            output_item = {
                "instruction": "Based on 2D Voxel scan, please predict objects in the 3D space in json format",
                "input": "<image>",
                "output": objects_str,
                "images": [item["image2d_path"]]
            }
            
            output_data.append(output_item)
            processed_count += 1
    
    # Write to output file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Conversion complete.")
    print(f"- Items processed: {processed_count}")
    print(f"- Items skipped (missing images): {skipped_count}")
    print(f"- Total items written to {output_file}: {len(output_data)}")

def main():
    # Check if source path is provided
    if len(sys.argv) < 2:
        print("Usage: python script.py <source_path>")
        sys.exit(1)

    # Get source path from command line argument
    source_path = sys.argv[1]
    
    # Define source and destination paths
    source_folder = os.path.join(source_path, "image2d")
    source_file = os.path.join(source_path, "labels.jsonl")
    
    data_dir = "./data"
    dest_folder = os.path.join(data_dir, "image2d")
    dest_file = os.path.join(data_dir, "labels.jsonl")
    
    # Create the data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Copy the image2d folder
    if os.path.exists(source_folder):
        print(f"Copying {source_folder} to {dest_folder}...")
        if os.path.exists(dest_folder):
            shutil.rmtree(dest_folder)  # Remove existing directory if it exists
        shutil.copytree(source_folder, dest_folder)
        print("Folder copied successfully.")
    else:
        print(f"Error: Source folder '{source_folder}' not found.")
        sys.exit(1)
    
    # Copy the labels.jsonl file
    if os.path.exists(source_file):
        print(f"Copying {source_file} to {dest_file}...")
        shutil.copy2(source_file, dest_file)
        print("File copied successfully.")
    else:
        print(f"Error: Source file '{source_file}' not found.")
        sys.exit(1)
    
    print("All files copied successfully to ./data directory.")

    # Example usage
    input_file = "data/labels.jsonl"  # Changed to labels.jsonl as mentioned
    output_file = "data/data.json"
    # Set image_dir to the base directory where your images are stored
    # If images are in the current directory, you can leave it empty
    image_dir = "data"  # For example: "/path/to/your/images" if needed

    convert_jsonl_to_json(input_file, output_file, image_dir)

if __name__ == "__main__":
    main()