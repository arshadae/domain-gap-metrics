"""
metrics/ssim.py

Compute SSIM between *paired* images in two directories.

- Pairs by filename (default), filename stem, or sorted order.
- Supports 'y' (luma), 'gray', or 'rgb' modes.
- Optionally resizes images from dir B to match dir A before SSIM.
- Returns mean ± std and number of pairs; can also write a CSV.

Requires: pillow, numpy, scikit-image
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import csv
import numpy as np
from PIL import Image

# skimage API changed at 0.19: channel_axis vs multichannel
from skimage.metrics import structural_similarity as ssim

__all__ = ["compute_ssim_dirs"]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ---------------- internal helpers ----------------

def _load_image(path: Path, mode: str) -> np.ndarray:
    """
    mode: 'y' (luma), 'gray', or 'rgb'
    Returns float64 array in [0,1] with shape HxW (gray) or HxWx3 (rgb).
    """
    im = Image.open(path).convert("RGB")
    if mode == "y":
        ycbcr = im.convert("YCbCr")
        y, _, _ = ycbcr.split()
        arr = np.asarray(y, dtype=np.float64) / 255.0  # HxW
        return arr
    elif mode == "gray":
        g = im.convert("L")
        arr = np.asarray(g, dtype=np.float64) / 255.0  # HxW
        return arr
    elif mode == "rgb":
        arr = np.asarray(im, dtype=np.float64) / 255.0  # HxWx3
        return arr
    else:
        raise ValueError("SSIM mode must be one of {'y','gray','rgb'}")

def _resize_to_match(b_img: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    if b_img.shape == shape:
        return b_img
    if b_img.ndim == 2:
        im = Image.fromarray(np.clip(b_img * 255.0, 0, 255).astype(np.uint8), mode="L")
        im = im.resize((shape[1], shape[0]), Image.BICUBIC)
        return np.asarray(im, dtype=np.float64) / 255.0
    else:
        im = Image.fromarray(np.clip(b_img * 255.0, 0, 255).astype(np.uint8), mode="RGB")
        im = im.resize((shape[1], shape[0]), Image.BICUBIC)
        return np.asarray(im, dtype=np.float64) / 255.0

def _compute_ssim_pair(a_img: np.ndarray, b_img: np.ndarray) -> float:
    """
    Calls skimage SSIM with Gaussian window (sigma=1.5), data_range=1.0.
    Handles both old and new skimage APIs.
    """
    if a_img.ndim != b_img.ndim:
        raise ValueError("SSIM: mismatched dims (RGB vs gray). Use same mode for both dirs.")
    channel_axis = -1 if a_img.ndim == 3 else None
    kwargs = dict(data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    try:
        # New API (>= 0.19)
        return float(ssim(a_img, b_img, channel_axis=channel_axis, **kwargs))
    except TypeError:
        # Old API (< 0.19)
        return float(ssim(a_img, b_img, multichannel=(channel_axis is not None), **kwargs))

def _pair_by_name(dir_a: Path, dir_b: Path) -> List[Tuple[Path, Path]]:
    b_map = {p.name: p for p in dir_b.rglob("*") if p.suffix.lower() in IMG_EXTS}
    pairs = []
    for pa in sorted([p for p in dir_a.rglob("*") if p.suffix.lower() in IMG_EXTS]):
        pb = b_map.get(pa.name)
        if pb:
            pairs.append((pa, pb))
    return pairs

def _pair_by_stem(dir_a: Path, dir_b: Path) -> List[Tuple[Path, Path]]:
    b_map = {p.stem: p for p in dir_b.rglob("*") if p.suffix.lower() in IMG_EXTS}
    pairs = []
    for pa in sorted([p for p in dir_a.rglob("*") if p.suffix.lower() in IMG_EXTS]):
        pb = b_map.get(pa.stem)
        if pb:
            pairs.append((pa, pb))
    return pairs

def _pair_by_sorted(dir_a: Path, dir_b: Path) -> List[Tuple[Path, Path]]:
    a_list = sorted([p for p in dir_a.rglob("*") if p.suffix.lower() in IMG_EXTS])
    b_list = sorted([p for p in dir_b.rglob("*") if p.suffix.lower() in IMG_EXTS])
    n = min(len(a_list), len(b_list))
    return list(zip(a_list[:n], b_list[:n]))


# ---------------- public API ----------------

def compute_ssim_dirs(
    dir_a: Path | str,
    dir_b: Path | str,
    *,
    pairing: str = "name",     # 'name' | 'stem' | 'sorted'
    mode: str = "y",           # 'y' | 'gray' | 'rgb'
    resize_b_to_a: bool = True,
    save_csv: Optional[Path | str] = None
) -> Dict[str, object]:
    """
    Compute SSIM over paired images from two directories.

    Returns:
        dict with keys: 'pairs', 'mean', 'std', 'mode', 'pairing'
        and writes an optional CSV with per-image SSIM.
    """
    dir_a, dir_b = Path(dir_a), Path(dir_b)
    if pairing == "name":
        pairs = _pair_by_name(dir_a, dir_b)
    elif pairing == "stem":
        pairs = _pair_by_stem(dir_a, dir_b)
    elif pairing == "sorted":
        pairs = _pair_by_sorted(dir_a, dir_b)
    else:
        raise ValueError("SSIM pairing must be one of {'name','stem','sorted'}")

    if not pairs:
        raise RuntimeError("SSIM: no pairs found. Check folders and pairing method.")

    scores, rows = [], []
    for pa, pb in pairs:
        a = _load_image(pa, mode)
        b = _load_image(pb, mode)
        if a.shape != b.shape:
            if not resize_b_to_a:
                raise ValueError(f"SSIM: size mismatch for {pa.name} vs {pb.name} and resize disabled.")
            b = _resize_to_match(b, a.shape)
        s = _compute_ssim_pair(a, b)
        scores.append(s)
        rows.append({"file": pa.name, "ssim": s})

    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0

    if save_csv:
        save_csv = Path(save_csv)
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(save_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file", "ssim"])
            w.writeheader()
            w.writerows(rows)

    return {"pairs": len(scores), "mean": mean, "std": std, "mode": mode, "pairing": pairing}


# Optional: run as a script
if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser(description="Compute SSIM between paired images in two directories.")
    ap.add_argument("--dir-a", required=True)
    ap.add_argument("--dir-b", required=True)
    ap.add_argument("--pairing", default="name", choices=["name", "stem", "sorted"])
    ap.add_argument("--mode", default="y", choices=["y", "gray", "rgb"])
    ap.add_argument("--no-resize", action="store_true")
    ap.add_argument("--save-csv", default=None)
    args = ap.parse_args()

    res = compute_ssim_dirs(
        args.dir_a, args.dir_b,
        pairing=args.pairing, mode=args.mode,
        resize_b_to_a=not args.no_resize,
        save_csv=args.save_csv
    )
    print(f"Pairs: {res['pairs']}")
    print(f"SSIM ({res['mode']} / {res['pairing']}): {res['mean']:.6f} ± {res['std']:.6f}")
    sys.exit(0)
