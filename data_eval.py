
import argparse
import json
import os
import math
from typing import Dict, List, Optional, Tuple

###############################################################################
# Helper functions
###############################################################################

def read_jsonl(path: str) -> Dict[int, dict]:
    """
    Reads a .jsonl file and returns a dictionary keyed by the top-level "id" 
    field. Each value is the parsed JSON object for that example.

    We assume each line in the JSONL file has the structure:
    {
        "id": int,
        "objects": [
            {
             "id": "...",
             "color": "...",
             "description": "...",
             "number_of_occupied_voxel": ...,
             "voxel_coords_center": {"x": ..., "y": ..., "z": ...}
            },
            ...
        ]
    }
    """
    data = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rec_id = record["id"]
            data[rec_id] = record
    return data


def euclidean_distance_3d(pt1: dict, pt2: dict) -> float:
    """
    Computes Euclidean distance between two 3D points:
    pt1 = {"x": ..., "y": ..., "z": ...}
    pt2 = {"x": ..., "y": ..., "z": ...}
    """
    return math.sqrt(
        (pt1["x"] - pt2["x"])**2 +
        (pt1["y"] - pt2["y"])**2 +
        (pt1["z"] - pt2["z"])**2
    )


def find_nearest_object(
    pred_obj: dict, label_objects: List[dict]
) -> Optional[int]:
    """
    Given a single predicted object (pred_obj), find the index of the label object
    in 'label_objects' that has the minimal 3D center distance to it.
    Returns None if 'label_objects' is empty.
    """
    if not label_objects:
        return None

    min_dist = float('inf')
    min_index = None
    pred_center = pred_obj.get("voxel_coords_center", {})
    for i, lbl_obj in enumerate(label_objects):
        label_center = lbl_obj.get("voxel_coords_center", {})
        # Validate presence of x,y,z on both
        if all(k in pred_center for k in ["x", "y", "z"]) and all(k in label_center for k in ["x", "y", "z"]):
            dist = euclidean_distance_3d(pred_center, label_center)
            if dist < min_dist:
                min_dist = dist
                min_index = i
        else:
            # If centers are missing, treat as infinite distance
            pass
        

    return min_index


def evaluate_example(label_example: dict, output_example: dict):
    """
    Match each predicted object with its nearest label object (greedy).  
    Compute various metrics on matched pairs, and track how many unmatched 
    objects remain. 
    Returns a dictionary of stats for this example.
    """
    label_objs = label_example.get("objects", [])
    pred_objs  = output_example.get("objects", [])

    # We'll copy the label_objs so we can remove matched ones
    unmatched_labels = label_objs.copy()
    unmatched_preds  = []

    # Keep track of matched pairs for stats
    center_dist_list       = []
    color_match_list       = []
    desc_match_list        = []
    voxel_diff_list        = []

    for pred_obj in pred_objs:
        idx = find_nearest_object(pred_obj, unmatched_labels)
        if idx is not None:
            label_obj = unmatched_labels[idx]
            # Compute stats
            dist = 0.0
            pred_center = pred_obj.get("voxel_coords_center", {})
            label_center = label_obj.get("voxel_coords_center", {})
            if all(k in pred_center for k in ["x", "y", "z"]) and all(k in label_center for k in ["x", "y", "z"]):
                dist = euclidean_distance_3d(pred_center, label_center)
            center_dist_list.append(dist)

            # color match
            pred_color  = pred_obj.get("color")
            label_color = label_obj.get("color")
            color_match_list.append(1 if pred_color == label_color else 0)

            # description match
            pred_desc  = pred_obj.get("description")
            label_desc = label_obj.get("description")
            desc_match_list.append(1 if pred_desc == label_desc else 0)

            # number_of_occupied_voxel difference
            pred_voxel_count  = pred_obj.get("number_of_occupied_voxel", 0)
            label_voxel_count = label_obj.get("number_of_occupied_voxel", 0)
            voxel_diff = abs(pred_voxel_count - label_voxel_count)
            voxel_diff_list.append(voxel_diff)

            # Remove this label from unmatched
            unmatched_labels.pop(idx)
        else:
            # no label available - this predicted object is unmatched
            unmatched_preds.append(pred_obj)

    # After matching is done:
    # "unmatched_labels" are label objects with no predicted match
    # "unmatched_preds"  are predicted objects with no label match
    mismatch_count = len(unmatched_labels) + len(unmatched_preds)

    n_matched = len(center_dist_list)  # number of matched pairs

    # Compute average stats for matched pairs
    avg_center_dist = (sum(center_dist_list) / n_matched) if n_matched > 0 else float('nan')
    color_acc = (sum(color_match_list) / n_matched) if n_matched > 0 else 0.0
    desc_acc  = (sum(desc_match_list) / n_matched)  if n_matched > 0 else 0.0
    avg_voxel_diff = (sum(voxel_diff_list) / n_matched) if n_matched > 0 else 0.0

    # Form example's stats
    example_stats = {
        "num_label_objs": len(label_objs),
        "num_pred_objs": len(pred_objs),
        "num_matched": n_matched,
        "mismatch_count": mismatch_count,
        "avg_center_distance": avg_center_dist,
        "color_accuracy": color_acc,
        "desc_accuracy": desc_acc,
        "avg_voxel_count_diff": avg_voxel_diff,
    }
    return example_stats


def evaluate_dataset(label_path: str, output_path: str) -> None:
    """
    Reads label.jsonl and output.jsonl, then for each example ID shared by 
    both files, does a greedy nearest-object matching to compute stats.  
    Prints per-example results and overall averages across the dataset.
    """
    label_data = read_jsonl(label_path)
    output_data = read_jsonl(output_path)

    # keys that exist in both
    example_ids = sorted(set(label_data.keys()) & set(output_data.keys()))

    # Accumulators for final overall metrics
    total_examples = 0
    sum_center_dist = 0.0
    sum_color_match = 0
    sum_desc_match  = 0
    sum_voxel_diff  = 0.0
    sum_mismatch    = 0

    count_matched_pairs = 0

    for ex_id in example_ids:
        lbl_example = label_data[ex_id]
        out_example = output_data[ex_id]

        stats = evaluate_example(lbl_example, out_example)

        # Accumulate for global stats
        total_examples += 1
        num_matched = stats["num_matched"]
        if num_matched > 0:
            sum_center_dist += stats["avg_center_distance"] * num_matched
            sum_color_match += stats["color_accuracy"] * num_matched
            sum_desc_match  += stats["desc_accuracy"] * num_matched
            sum_voxel_diff  += stats["avg_voxel_count_diff"] * num_matched
            count_matched_pairs += num_matched

        sum_mismatch += stats["mismatch_count"]

    # Compute overall stats
    if total_examples == 0:
        print("No overlapping examples to evaluate.")
        return

    # Averages across all matched object pairs in the dataset
    if count_matched_pairs == 0:
        overall_avg_center_dist = float('nan')
        overall_color_acc = 0.0
        overall_desc_acc  = 0.0
        overall_voxel_diff = 0.0
    else:
        overall_avg_center_dist = sum_center_dist / count_matched_pairs
        overall_color_acc = sum_color_match / count_matched_pairs
        overall_desc_acc  = sum_desc_match / count_matched_pairs
        overall_voxel_diff = sum_voxel_diff / count_matched_pairs

    # mismatch_count across examples
    # This metric is “how many objects ended up unmatched in total,” 
    # but you could also define it as an average per example, etc.
    overall_mismatch_count = sum_mismatch
    avg_mismatch_per_example = sum_mismatch / total_examples

    print("================================================")
    print("Summary across all examples:")
    print(f"  - total examples: {total_examples}")
    print(f"  - total matched pairs: {count_matched_pairs}")
    print(f"  - overall avg center distance (matched only): {overall_avg_center_dist:.4f}")
    print(f"  - overall color accuracy (matched only): {overall_color_acc:.2f}")
    print(f"  - overall desc accuracy (matched only): {overall_desc_acc:.2f}")
    print(f"  - overall avg voxel count diff (matched only): {overall_voxel_diff:.2f}")
    print(f"  - total unmatched objects (mismatch_count): {overall_mismatch_count}")
    print(f"  - avg mismatch per example: {avg_mismatch_per_example:.2f}")


###############################################################################
# CLI interface
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictions vs. labels using nearest-object match.")
    parser.add_argument("--label", "-l", type=str, required=True, help="Path to label.jsonl")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output.jsonl")
    args = parser.parse_args()

    if not os.path.isfile(args.label):
        raise FileNotFoundError(f"Label file not found: {args.label}")
    if not os.path.isfile(args.output):
        raise FileNotFoundError(f"Output file not found: {args.output}")

    evaluate_dataset(args.label, args.output)