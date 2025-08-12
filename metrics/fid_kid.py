import numpy as np
from scipy import linalg

def compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm((sigma1 + eps*np.eye(sigma1.shape[0])) @ (sigma2 + eps*np.eye(sigma2.shape[0])), disp=False)
    if np.iscomplexobj(covmean): covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean))

def compute_kid(feats1, feats2, n_subsets=100, subset_size=1000):
    n1, n2, d = feats1.shape[0], feats2.shape[0], feats1.shape[1]
    subset_size = min(subset_size, n1, n2)
    vals = []
    for _ in range(n_subsets):
        x = feats1[np.random.choice(n1, subset_size, replace=False)]
        y = feats2[np.random.choice(n2, subset_size, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        np.fill_diagonal(kxx, 0); np.fill_diagonal(kyy, 0)
        m = subset_size
        vals.append(kxx.sum()/(m*(m-1)) + kyy.sum()/(m*(m-1)) - 2*kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))
