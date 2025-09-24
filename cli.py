#!/usr/bin/env python3

import argparse

def build_parser():
    p = argparse.ArgumentParser(description="Domain gap evaluation for two image folders.")
    p.add_argument("--domain_a", type=str, required=True)
    p.add_argument("--domain_b", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./domain_gap_reports")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--lpips", action="store_true")
    p.add_argument("--lpips_pairs", type=int, default=50)
    p.add_argument("--eval_name", type=str, default=None)
    p.add_argument("--task", choices=["classification", "segmentation", "detection"])
    p.add_argument("--model_py", type=str)
    p.add_argument("--out_json_suffix", type=str, default=None)
    # classification
    p.add_argument("--labels_csv", type=str)
    # segmentation
    p.add_argument("--mask_dir", type=str)
    p.add_argument("--num_classes", type=int)
    p.add_argument("--ignore_index", type=int, default=None)
    p.add_argument("--mask_suffix", type=str, default=".png")
    # detection
    p.add_argument("--coco_gt", type=str)
    p.add_argument("--score_thresh", type=float, default=0.0)
    return p

def main(argv=None):
    from orchestrator import run
    
    args = build_parser().parse_args(argv)
    run(args)



if __name__ == "__main__":
    main()
