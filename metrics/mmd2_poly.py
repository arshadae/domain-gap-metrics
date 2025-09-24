"""
Kernel Inception Distance (KID) — Binkowski et al., 2018 (ICLR)
Unbiased MMD^2 on Inception features with a polynomial kernel.

- Default kernel: degree=3, coef=1.0, gamma=1/d (d = feature dim)
- Report mean ± std over random subsets (as in the paper)
"""

from typing import Tuple, Optional, Union
import numpy as np
import torch

@torch.no_grad()
def kid_from_features_mmdgan(
    feats_real: Union[np.ndarray, torch.Tensor],
    feats_fake: Union[np.ndarray, torch.Tensor],
    *,
    subsets: int = 100,
    subset_size: int = 1000,
    seed: Optional[int] = 123,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> Tuple[float, float]:
    X = torch.as_tensor(feats_real, dtype=dtype, device=device)
    Y = torch.as_tensor(feats_fake, dtype=dtype, device=device)

    n, m = X.size(0), Y.size(0)
    s = int(min(subset_size, n, m))
    if s < 2:
        raise ValueError("subset_size must be ≥2 and ≤ min(N_real, N_fake).")

    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)

    vals = []
    for _ in range(subsets):
        ix = torch.randperm(n, generator=g, device=device)[:s]
        iy = torch.randperm(m, generator=g, device=device)[:s]
        Xs = X.index_select(0, ix)      # [s, d]
        Ys = Y.index_select(0, iy)      # [s, d]
        m_ = s
        d_ = Xs.size(1)

        Kxx = (Xs @ Xs.T) / d_ + 1.0
        Kyy = (Ys @ Ys.T) / d_ + 1.0
        Kxy = (Xs @ Ys.T) / d_ + 1.0

        a = Kxx.pow(3) + Kyy.pow(3)
        b = Kxy.pow(3)
        kid_subset = ((a.sum() - torch.diagonal(a).sum()) / (m_ - 1)
                      - 2.0 * b.sum() / m_) / m_
        vals.append(kid_subset.item())

    vals = torch.tensor(vals, dtype=dtype).cpu().numpy()
    return float(vals.mean()), float(vals.std(ddof=0))


# ---------------------------
# Example: using pytorch-fid to get features, then compute KID
# ---------------------------
if __name__ == "__main__":
    # 1) Extract Inception-V3 pool3 features with pytorch-fid
    #    (pip install pytorch-fid)
    from pytorch_fid.fid_score import get_activations
    from pytorch_fid.inception import InceptionV3

    # Replace with your folders
    paths_real = ["/path/to/real_images"]
    paths_fake = ["/path/to/generated_images"]

    dims = 2048  # pool3
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception = InceptionV3([block_idx]).to(device)

    feats_real = get_activations(paths_real, inception, batch_size, dims, device, num_workers=4)
    feats_fake = get_activations(paths_fake, inception, batch_size, dims, device, num_workers=4)

    # 2) Compute KID (paper defaults: degree=3, coef=1, gamma=1/d)
    kid_mean, kid_std = kid_from_features_mmdgan(
        feats_real, feats_fake,
        subsets=100, subset_size=1000, seed=123, device=None
    )
    print(f"KID: {kid_mean:.6f} ± {kid_std:.6f}")
    print(f"KID: {kid_mean:.6f} ± {kid_std:.6f}")
