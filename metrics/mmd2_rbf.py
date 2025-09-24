import torch
import numpy as np
from typing import Iterable, Optional, Tuple, Union

# ---------- helpers ----------
def _pairwise_sq_dists(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    # Efficient squared Euclidean distances
    X2 = (X * X).sum(dim=1, keepdim=True)        # [n,1]
    Y2 = (Y * Y).sum(dim=1, keepdim=True).T      # [1,m]
    D = X2 + Y2 - 2.0 * (X @ Y.T)
    return D.clamp_min_(0)

@torch.no_grad()
def _mmd2_rbf_unbiased_single_sigma(X: torch.Tensor, Y: torch.Tensor, sigma: float) -> torch.Tensor:
    n, m = X.size(0), Y.size(0)
    if n < 2 or m < 2:
        raise ValueError("Need at least 2 samples per set for unbiased MMD^2.")

    Dxx = _pairwise_sq_dists(X, X)
    Dyy = _pairwise_sq_dists(Y, Y)
    Dxy = _pairwise_sq_dists(X, Y)

    Kxx = torch.exp(-Dxx / (2.0 * sigma * sigma))
    Kyy = torch.exp(-Dyy / (2.0 * sigma * sigma))
    Kxy = torch.exp(-Dxy / (2.0 * sigma * sigma))

    # unbiased: remove self-similarity
    Kxx.fill_diagonal_(0)
    Kyy.fill_diagonal_(0)

    term_x = Kxx.sum() / (n * (n - 1))
    term_y = Kyy.sum() / (m * (m - 1))
    term_xy = 2.0 * Kxy.mean()
    return term_x + term_y - term_xy

def _median_heuristic_sigma(X: torch.Tensor, Y: torch.Tensor, max_samples: int = 2000) -> float:
    # Estimate bandwidth via median pairwise distance on a subset
    n = X.size(0) + Y.size(0)
    idx = torch.randperm(n)[:min(n, max_samples)]
    Z = torch.cat([X, Y], dim=0).index_select(0, idx)
    D = _pairwise_sq_dists(Z, Z)
    # take upper-triangular (exclude zeros on diagonal)
    tri = D[torch.triu(torch.ones_like(D, dtype=torch.bool), diagonal=1)]
    med = torch.median(tri[tri > 0])
    # Convert to sigma for exp(-||x-y||^2/(2*sigma^2))
    sigma = torch.sqrt(med / 2.0).item() if med > 0 else 1.0
    return float(sigma)

# ---------- public API ----------
@torch.no_grad()
def mmd2_rbf_from_features(
    feats_real: Union[np.ndarray, torch.Tensor],
    feats_fake: Union[np.ndarray, torch.Tensor],
    *,
    sigmas: Union[str, Iterable[float]] = "auto-median-x{0.5,1,2,4,8}",
    weights: Optional[Iterable[float]] = None,
    subsets: int = 1,
    subset_size: Optional[int] = None,
    seed: Optional[int] = 123,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> Tuple[float, Optional[float]]:
    """
    Compute squared MMD with an RBF kernel on feature sets.
    - Unbiased estimator (diagonals removed).
    - Multi-scale: average (or weighted average) across bandwidths.

    Args:
        feats_real, feats_fake: [N, D] arrays/tensors (e.g., Inception pool3).
        sigmas:
            - list/tuple of bandwidths (in feature units), OR
            - "auto-median"  -> one sigma via median heuristic, OR
            - "auto-median-x{...}" -> ladder around median, e.g. "auto-median-x{0.5,1,2,4}"
        weights: optional weights for each sigma (defaults to uniform).
        subsets: number of random subsets to average (≥1). Like KID, improves stability.
        subset_size: per-subset sample size; clipped to min(N_real, N_fake). If None, use all.
        seed: RNG seed for reproducibility (sampling only).
        device/dtype: compute placement and precision (float64 recommended).

    Returns:
        (mmd2_mean, mmd2_std or None)
    """
    X = torch.as_tensor(feats_real, dtype=dtype, device=device)
    Y = torch.as_tensor(feats_fake, dtype=dtype, device=device)

    n, m = X.size(0), Y.size(0)
    s = min(n, m) if subset_size is None else min(subset_size, n, m)
    if s < 2:
        raise ValueError("subset_size must be ≥2 and ≤ min(N_real, N_fake).")

    # bandwidths
    if isinstance(sigmas, str):
        if sigmas == "auto-median":
            sigma0 = _median_heuristic_sigma(X, Y)
            sigma_list = [sigma0]
        elif sigmas.startswith("auto-median-x"):
            # parse multipliers, e.g. "auto-median-x{0.5,1,2,4,8}"
            inside = sigmas[sigmas.find("{")+1 : sigmas.find("}")]
            multipliers = [float(t.strip()) for t in inside.split(",")]
            sigma0 = _median_heuristic_sigma(X, Y)
            sigma_list = [sigma0 * c for c in multipliers]
        else:
            raise ValueError(f"Unrecognized sigmas spec: {sigmas}")
    else:
        sigma_list = list(float(s) for s in sigmas)

    if weights is None:
        w = torch.ones(len(sigma_list), dtype=dtype, device=device) / len(sigma_list)
    else:
        w = torch.as_tensor(list(weights), dtype=dtype, device=device)
        w = w / w.sum()

    # sampling setup (for subset averaging)
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)

    vals = []
    for _ in range(max(1, subsets)):
        if s < n:
            ix = torch.randperm(n, generator=g, device=device)[:s]
            Xs = X.index_select(0, ix)
        else:
            Xs = X
        if s < m:
            iy = torch.randperm(m, generator=g, device=device)[:s]
            Ys = Y.index_select(0, iy)
        else:
            Ys = Y

        per_sigma = []
        for sigma in sigma_list:
            per_sigma.append(_mmd2_rbf_unbiased_single_sigma(Xs, Ys, sigma))
        per_sigma = torch.stack(per_sigma)            # [S]
        vals.append((w * per_sigma).sum().item())     # weighted average across sigmas

    vals = np.asarray(vals, dtype=float)
    mean = float(vals.mean())
    std = float(vals.std(ddof=0)) if subsets > 1 else None
    return mean, std
