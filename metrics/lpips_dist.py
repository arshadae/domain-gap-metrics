# metrics/lpips_dist.py
from __future__ import annotations
from typing import List, Optional, Sequence, Union
from pathlib import Path
import torch, lpips
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

__all__ = ["compute_lpips_set_distance"]

# Load a batch of images (by explicit indices) and normalize to [-1, 1]
def _load_batch_by_indices(paths: Sequence[Union[str, Path]],
                           indices: Sequence[int],
                           resize_to: Optional[int]) -> torch.Tensor:
    imgs = []
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize(resize_to, interpolation=InterpolationMode.BICUBIC) if resize_to else None
    for idx in indices:
        p = paths[idx]
        img = Image.open(p).convert("RGB")
        if resize:
            img = resize(img)
        x = to_tensor(img) * 2.0 - 1.0  # [-1,1] required by official LPIPS
        imgs.append(x)
    return torch.stack(imgs, dim=0)  # [B,C,H,W]

@torch.no_grad()
def compute_lpips_set_distance(
    paths_a: Sequence[Union[str, Path]],
    paths_b: Sequence[Union[str, Path]],
    *,
    sample_pairs: Optional[int] = None,   # None => index-paired mean; int => random pairs
    device: str = "cpu",
    net: str = "alex",                    # {"alex","vgg","squeeze"}; "alex" is the common default
    resize_to: Optional[int] = 256,       # ensure same HxW; set None only if all images already match
    batch_size: int = 32,
    seed: Optional[int] = 123,
) -> float:
    """
    Compute a set-to-set LPIPS distance between two folders’ images.

    If sample_pairs is None:
        - Sort both lists, pair by index, truncate to min length, and average.
    If sample_pairs is an int:
        - Draw that many random pairs (with replacement) across both sets and average.

    Returns:
        Mean LPIPS across the chosen pairs (float).
    """
    # Normalize inputs to strings for PIL
    A: List[str] = [str(p) for p in paths_a]
    B: List[str] = [str(p) for p in paths_b]
    if len(A) == 0 or len(B) == 0:
        raise ValueError("compute_lpips_set_distance: one of the folders is empty.")

    model = lpips.LPIPS(net=net).to(device).eval()

    if sample_pairs is None:
        A_sorted = sorted(A)
        B_sorted = sorted(B)
        n = min(len(A_sorted), len(B_sorted))
        if n == 0:
            raise ValueError("No overlapping pairs to compare.")
        total = 0.0
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            A_batch = _load_batch_by_indices(A_sorted, range(i, j), resize_to).to(device)
            B_batch = _load_batch_by_indices(B_sorted, range(i, j), resize_to).to(device)
            d = model(A_batch, B_batch).view(-1)
            total += float(d.sum().item())
        return total / n

    # Random-pair mode (with replacement)
    import random
    rng = random.Random(seed)
    total = 0.0
    remaining = int(sample_pairs)
    while remaining > 0:
        bs = min(batch_size, remaining)
        idx_a = [rng.randrange(len(A)) for _ in range(bs)]
        idx_b = [rng.randrange(len(B)) for _ in range(bs)]
        A_batch = _load_batch_by_indices(A, idx_a, resize_to).to(device)
        B_batch = _load_batch_by_indices(B, idx_b, resize_to).to(device)
        d = model(A_batch, B_batch).view(-1)
        total += float(d.sum().item())
        remaining -= bs
    return total / float(sample_pairs)
