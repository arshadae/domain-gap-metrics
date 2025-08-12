# domain_gap_eval/tasks/classification.py
import numpy as np
import csv
from pathlib import Path
from typing import List
import torch
from tqdm import tqdm
from PIL import Image

def pil_loader(path: Path):
    with Image.open(path) as im:
        return im.convert("RGB")

def _confusion_and_prf(y_true: np.ndarray, y_pred: np.ndarray):
    C = int(max(y_true.max(), y_pred.max())) + 1 if len(y_true) else 0
    conf = np.zeros((C, C), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1

    precision, recall, f1 = [], [], []
    for c in range(C):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        f = 2 * p * r / (p + r + 1e-12)
        precision.append(float(p))
        recall.append(float(r))
        f1.append(float(f))

    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    return conf, precision, recall, f1, acc

def _read_label_csv(img_dir: str, labels_csv: str):
    paths: List[Path] = []
    y_true: List[int] = []
    with open(labels_csv, "r") as f:
        reader = csv.DictReader(f)
        if "filename" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("labels_csv must have columns: filename,label")
        for row in reader:
            p = Path(img_dir) / row["filename"]
            if p.exists():
                paths.append(p)
                y_true.append(int(row["label"]))
    if not paths:
        raise RuntimeError("No labeled images matched filenames in labels_csv.")
    return paths, np.array(y_true, dtype=np.int64)

def evaluate_classification(img_dir: str,
                            labels_csv: str,
                            model,
                            transform,
                            device: str,
                            batch_size: int = 32):
    """
    Supports two modes:
      1) Standard PyTorch classifier: logits = model(x)
      2) Filename-aware dummy: if model has `predict_pil(pil_images, filenames)`,
         we call that to produce logits where argmax is the predicted class.
    """
    # Load file list + labels
    paths, y_true = _read_label_csv(img_dir, labels_csv)

    # Inference
    preds = []
    for i in tqdm(range(0, len(paths), batch_size), desc="Cls inference", leave=False):
        batch_paths = paths[i:i+batch_size]
        pil_imgs = [pil_loader(p) for p in batch_paths]

        with torch.no_grad():
            if hasattr(model, "predict_pil"):
                # Filename-aware path (dummy/perfect classifier use-case)
                logits = model.predict_pil(pil_imgs, [p.name for p in batch_paths])
                if not isinstance(logits, torch.Tensor):
                    logits = torch.as_tensor(logits)
            else:
                # Standard classifier path
                x = torch.stack([transform(im) for im in pil_imgs]).to(device)
                logits = model(x)

        y_hat = logits.argmax(1).cpu().numpy()
        preds.append(y_hat)

    y_pred = np.concatenate(preds, axis=0)

    # Metrics
    conf, precision, recall, f1, acc = _confusion_and_prf(y_true, y_pred)

    return {
        "task": "classification",
        "accuracy": acc,
        "per_class": {"precision": precision, "recall": recall, "f1": f1},
        "confusion_matrix": conf.tolist(),
        "num_samples": int(len(y_true)),
        "num_classes": int(conf.shape[0]),
    }
