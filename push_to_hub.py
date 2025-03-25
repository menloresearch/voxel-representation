import argparse
from huggingface_hub import login, create_repo, upload_folder

def main(username: str, checkpoint_num: str, path_to_model_checkpoint: str):
    # login()
    
    repo_id = f"{username}/voxel-representation-gemma3-4b-cp-{checkpoint_num}"
    create_repo(repo_id, repo_type="model", exist_ok=True, private=True)
    
    folder_path = f"{path_to_model_checkpoint}/checkpoint-{checkpoint_num}"
    

    # Specify the patterns you'd like to ignore below.
    # For example, ignore the entire "global_step*" folder and
    # any bf16_zero_pp_rank_*.pt files:
    ignore_list = [
        "global_step*",              # This will ignore any directory or file starting with 'global_step'
        "bf16_zero_pp_rank_*.pt"     # Or adapt to match the actual pattern of the files you want to skip
    ]


    upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload checkpoint-{checkpoint_num}",
        ignore_patterns=ignore_list
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a specific checkpoint to Hugging Face")
    parser.add_argument("username", type=str, help="Hugging Face username")
    parser.add_argument("checkpoint_num", type=str, help="Checkpoint number")
    parser.add_argument("path_to_model_checkpoint", type=str, help="path to the model checkpoint")
    args = parser.parse_args()
    
    main(args.username, args.checkpoint_num, args.path_to_model_checkpoint)
