import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
from PIL import Image

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}

def pil_loader(path):
    with Image.open(path) as im: return im.convert("RGB")

def _fast_hist(pred, gt, C, ignore_index=None):
    mask = (gt >= 0) & (gt < C)
    if ignore_index is not None: mask &= (gt != ignore_index)
    return np.bincount(C*gt[mask].astype(int)+pred[mask].astype(int), minlength=C*C).reshape(C, C)

def evaluate_segmentation(img_dir, mask_dir, model, transform, device,
                          num_classes, ignore_index=None, mask_suffix=".png", batch_size=4):
    img_paths = [p for p in sorted(Path(img_dir).rglob("*")) if p.suffix.lower() in IMG_EXTS]
    if not img_paths: raise RuntimeError("No images for segmentation.")
    hist = np.zeros((num_classes, num_classes), dtype=np.int64); total_pix = correct_pix = 0

    for i in tqdm(range(0, len(img_paths), batch_size), desc="Seg inference", leave=False):
        batch = img_paths[i:i+batch_size]
        ims = [pil_loader(p) for p in batch]
        x = torch.stack([transform(im) for im in ims]).to(device)
        with torch.no_grad(): logits = model(x)  # (B,C,H,W)
        pred = logits.argmax(1).cpu().numpy()
        for b, p in enumerate(batch):
            gt_path = Path(mask_dir) / (p.stem + mask_suffix)
            if not gt_path.exists(): continue
            gt = np.array(Image.open(gt_path), dtype=np.int64)
            ph, pw = pred[b].shape
            if gt.shape != (ph, pw):
                gt = np.array(Image.fromarray(gt.astype(np.uint8)).resize((pw, ph), resample=Image.NEAREST), dtype=np.int64)
            hist += _fast_hist(pred[b], gt, num_classes, ignore_index)
            if ignore_index is not None:
                m = (gt != ignore_index); correct_pix += int(((pred[b] == gt) & m).sum()); total_pix += int(m.sum())
            else:
                correct_pix += int((pred[b] == gt).sum()); total_pix += gt.size

    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-12)
    return {
        "task": "segmentation",
        "mean_IoU": float(np.nanmean(iu)),
        "per_class_IoU": iu.tolist(),
        "pixel_accuracy": float(correct_pix / (total_pix + 1e-12)),
        "num_images": int(len(img_paths)),
        "num_classes": int(num_classes),
        "ignore_index": (None if ignore_index is None else int(ignore_index)),
    }
