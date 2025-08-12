#!/usr/bin/env python3
"""
Synthetic dataset generator for testing evaluation pipelines.
Generates:
  - Classification dataset + labels CSV
  - Segmentation dataset (images + masks)
  - Detection dataset in COCO format
"""

import numpy as np
from PIL import Image
from pathlib import Path
import json
import random
import argparse


def generate_classification(root: Path, num_images: int, num_classes: int):
    cls_dir = root / "cls"
    cls_dir.mkdir(parents=True, exist_ok=True)
    labels_csv = root / "cls_labels.csv"

    with open(labels_csv, "w") as f:
        f.write("filename,label\n")
        for i in range(num_images):
            arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            im = Image.fromarray(arr)
            fname = f"img_{i}.png"
            im.save(cls_dir / fname)
            f.write(f"{fname},{random.randint(0, num_classes-1)}\n")
    print(f"[OK] Classification dataset in {cls_dir}, labels in {labels_csv}")


def generate_segmentation(root: Path, num_images: int, num_classes: int):
    seg_img_dir = root / "seg_images"
    seg_mask_dir = root / "seg_masks"
    seg_img_dir.mkdir(parents=True, exist_ok=True)
    seg_mask_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_images):
        arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        mask = np.random.randint(0, num_classes, (64, 64), dtype=np.uint8)
        Image.fromarray(arr).save(seg_img_dir / f"img_{i}.png")
        Image.fromarray(mask).save(seg_mask_dir / f"img_{i}.png")

    print(f"[OK] Segmentation dataset in {seg_img_dir} and {seg_mask_dir}")


def generate_detection(root: Path, num_images: int, num_classes: int):
    det_img_dir = root / "det_images"
    det_img_dir.mkdir(parents=True, exist_ok=True)
    coco_json_path = root / "det_coco.json"

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": cid, "name": f"class_{cid}"} for cid in range(1, num_classes+1)]
    }
    ann_id = 1
    for i in range(num_images):
        arr = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        fname = f"img_{i}.png"
        Image.fromarray(arr).save(det_img_dir / fname)
        coco["images"].append({"id": i, "file_name": fname, "width": 128, "height": 128})
        # random boxes
        for _ in range(random.randint(1, 3)):
            x, y = random.randint(0, 64), random.randint(0, 64)
            w, h = random.randint(10, 50), random.randint(10, 50)
            coco["annotations"].append({
                "id": ann_id,
                "image_id": i,
                "category_id": random.randint(1, num_classes),
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1

    with open(coco_json_path, "w") as f:
        json.dump(coco, f)

    print(f"[OK] Detection dataset in {det_img_dir}, annotations in {coco_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic datasets for eval testing.")
    parser.add_argument("--output_dir", type=str, default="synthetic_data", help="Root output dir")
    parser.add_argument("--num_images", type=int, default=20, help="Images per dataset")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes")
    args = parser.parse_args()

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    np.random.seed(42)

    generate_classification(root, args.num_images, args.num_classes)
    generate_segmentation(root, args.num_images, args.num_classes)
    generate_detection(root, args.num_images, args.num_classes)

    print(f"\nAll synthetic datasets generated in: {root.resolve()}")
