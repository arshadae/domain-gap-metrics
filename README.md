
# Domain Gap Evaluation Toolkit

A modular toolkit for **measuring domain gap** between two datasets using feature-level metrics (FID, KID, PAD, MMD, CKA, LPIPS, Histogram Distance) and task-specific metrics (classification accuracy/F1, segmentation IoU, detection mAP) for:
- **Image Classification**
- **Semantic Segmentation**
- **Object Detection**

The toolkit supports:
- Visualizations via **t-SNE** and **UMAP**
- Flexible model definitions via external Python scripts
- COCO-style detection evaluation
- Custom datasets with a plug-and-play interface

---

## Dataset
We include a **dummy dataset** (`dummy_data/`) for quick testing.

**Folder Structure:**
```
domain_gap_eval/
├── cli.py # Command-line entry point
├── orchestrator.py # Main orchestration logic
├── metrics/ # Feature-level metric implementations
├── tasks/ # Task-specific evaluation scripts
│ ├── classification.py
│ ├── segmentation.py
│ ├── detection.py
├── model_defs/ # Example PyTorch model definitions
├── dummy_data/ # Dummy datasets for testing
│ ├── seg_images/
│ ├── seg_masks/
│ ├── cls/
│ ├── cls_labels.csv
│ ├── det_images/
│ ├── det_coco.json
└── reports/ # Generated evaluation results & plots
```

---

## Installation

```bash
pip install -r requirements.txt
```

Make sure your `PYTHONPATH` is set to the repo root:
```bash
export PYTHONPATH=$(pwd)
```

---

## Usage

### 1. Classification Evaluation
```bash
python cli.py \
  --domain_a ./dummy_data/cls \
  --domain_b ./dummy_data/cls \
  --output_dir ./reports \
  --eval_name dummy_cls_test \
  --task classification \
  --model_py ./model_defs/model_cls.py \
  --labels_csv ./dummy_data/cls_labels.csv
```

---

### 2. Segmentation Evaluation
```bash
python cli.py \
  --domain_a ./dummy_data/seg_images \
  --domain_b ./dummy_data/seg_images \
  --output_dir ./reports \
  --eval_name dummy_seg_test \
  --task segmentation \
  --model_py ./model_defs/model_seg.py \
  --mask_dir ./dummy_data/seg_masks \
  --num_classes 5
```

---

### 3. Detection Evaluation

```bash
DGE_COCO_GT=./dummy_data/det_coco.json \
DGE_IMG_DIR=./dummy_data/det_images \
python cli.py \
  --domain_a ./dummy_data/det_images \
  --domain_b ./dummy_data/det_images \
  --output_dir ./reports \
  --eval_name dummy_det_test_gt \
  --task detection \
  --model_py ./model_defs/model_det.py \
  --coco_gt ./dummy_data/det_coco.json \
  --score_thresh 0.0
```
### 4. Task irrelevant evaluations
```bash 
python cli.py \
  --domain_a ./dummy_data/det_images \
  --domain_b ./dummy_data/det_images \
  --output_dir ./reports \
  --eval_name det_nogt_domain_only
  ```

### 5. Bash Script
```bash
run_eval.sh classification
run_eval.sh segmentation
run_eval.sh detection
```

---

## Metrics Computed

### **Feature-level metrics**
- **FID** – Fréchet Inception Distance
- **KID** – Kernel Inception Distance
- **PAD** – Proxy A-distance
- **MMD** – Maximum Mean Discrepancy
- **CKA** – Centered Kernel Alignment
- **LPIPS** – Perceptual similarity
- **Chi-Square RGB Histogram**

### **Task-specific metrics**
- **Classification:** Accuracy, Precision, Recall, F1-score, Confusion matrix
- **Segmentation:** mean IoU, per-class IoU, pixel accuracy
- **Detection:** COCO-style mAP and AR metrics

---

## Adapting to Your Own Dataset

To use your dataset:

### **For Classification:**
- Prepare two folders for `domain_a` and `domain_b` images.
- Provide a CSV file mapping filenames to class labels (format below).
- Create a .py file inside model_definitions/ that defines a get_model() function returning a PyTorch model.
- Run the evaluation with the same commands shown above, updating the dataset paths.
```python-repl
<!-- labels_csv file format -->
filename,label
image1.jpg,0
image2.jpg,1
...
```


### **For Segmentation:**
- Prepare two image folders for `domain_a` and `domain_b`.
- Provide a mask folder for ground-truth segmentation maps.
- Set `--num_classes` to match your dataset.
- Create a .py file inside model_definitions/ that defines a get_model() function returning a PyTorch model.
- Run the evaluation with the same commands shown above, updating the dataset paths.

### **For Detection:**
- Prepare two image folders for `domain_a` and `domain_b`.
- Provide a COCO-format JSON file with annotations.
- Create a .py file inside model_definitions/ that defines a get_model() function returning a PyTorch model.
- Run the evaluation with the same commands shown above, updating the dataset paths.
---

## Visualization
UMAP and t-SNE plots are saved in `reports/` to visualize domain gap.

---

## Notes
- Feature-level metrics alone do not guarantee equivalent task performance.
- Always combine **feature metrics** with **task metrics** for reliable domain gap assessment.
- Small synthetic datasets may give misleadingly high or low scores.

