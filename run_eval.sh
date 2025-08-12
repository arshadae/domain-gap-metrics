#!/bin/bash

# Usage:
#   ./run_eval.sh <task> [extra args...]
# Example:
#   ./run_eval.sh segmentation
#   ./run_eval.sh classification --labels_csv ./domain_gap_eval/dummy_data/cls_labels.csv
#   ./run_eval.sh detection --coco_gt ./domain_gap_eval/dummy_data/det_coco.json

TASK=$1
shift

COMMON_ARGS="--domain_a ./domain_gap_eval/dummy_data/${TASK}_images \
             --domain_b ./domain_gap_eval/dummy_data/${TASK}_images \
             --output_dir ./domain_gap_eval/reports \
             --eval_name dummy_${TASK}_test \
             --task $TASK"

if [ "$TASK" == "segmentation" ]; then
    python domain_gap_eval/cli.py $COMMON_ARGS \
        --model_py ./domain_gap_eval/model_defs/model_seg.py \
        --mask_dir ./domain_gap_eval/dummy_data/seg_masks \
        --num_classes 5 "$@"

elif [ "$TASK" == "classification" ]; then
    python domain_gap_eval/cli.py $COMMON_ARGS \
        --model_py ./domain_gap_eval/model_defs/model_cls.py \
        --labels_csv ./domain_gap_eval/dummy_data/cls_labels.csv "$@"

elif [ "$TASK" == "detection" ]; then
    python domain_gap_eval/cli.py $COMMON_ARGS \
        --model_py ./domain_gap_eval/model_defs/model_det.py \
        --coco_gt ./domain_gap_eval/dummy_data/det_coco.json "$@"

else
    echo "Unknown task: $TASK"
    echo "Valid options: segmentation | classification | detection"
    exit 1
fi
