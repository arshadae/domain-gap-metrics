#!/bin/bash

# Usage:
#   ./run_eval.sh <task> [extra args...]
# Example:
#   ./run_eval.sh segmentation
#   ./run_eval.sh classification --labels_csv ./domain_gap_eval/dummy_data/cls_labels.csv
#   ./run_eval.sh detection --coco_gt ./domain_gap_eval/dummy_data/det_coco.json

TASK=$1
shift
export DGE_IMG_DIR="${DGE_IMG_DIR:-$PWD/dummy_data/det_images}"
export DGE_COCO_GT="${DGE_COCO_GT:-$PWD/dummy_data/det_coco.json}"

COMMON_ARGS="--output_dir ./reports \
             --eval_name dummy_${TASK}_test \
             --task $TASK"

if [ "$TASK" == "segmentation" ]; then
    python cli.py $COMMON_ARGS \
        --domain_a ./dummy_data/seg_images \
        --domain_b ./dummy_data/seg_images \
        --model_py ./model_defs/model_seg.py \
        --mask_dir ./dummy_data/seg_masks \
        --num_classes 5 "$@"

elif [ "$TASK" == "classification" ]; then
    python cli.py $COMMON_ARGS \
        --domain_a ./dummy_data/cls \
        --domain_b ./dummy_data/cls \
        --model_py ./model_defs/model_cls.py \
        --labels_csv ./dummy_data/cls_labels.csv "$@"

elif [ "$TASK" == "detection" ]; then
    python cli.py $COMMON_ARGS \
        --domain_a ./dummy_data/det_images \
        --domain_b ./dummy_data/det_images \
        --model_py ./model_defs/model_det.py \
        --coco_gt ./dummy_data/det_coco.json \
        --score_thresh 0.0 "$@"

else
    echo "Unknown task: $TASK"
    echo "Valid options: segmentation | classification | detection"
    exit 1
fi
