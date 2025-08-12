import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap; HAS_UMAP=True
except Exception:
    HAS_UMAP=False
from pathlib import Path
from ..config import slugify
from packaging import version
import sklearn

def save_embedding_plot(feats_a, feats_b, out_png, method="tsne", name_suffix=None):
    X = np.vstack([feats_a, feats_b])
    y = np.array([0]*len(feats_a) + [1]*len(feats_b))
    Xp = PCA(n_components=min(50, X.shape[1]), random_state=42).fit_transform(X)
    if method == "umap" and HAS_UMAP:
        Z = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1).fit_transform(Xp)
    else:
        kws = dict(n_components=2, init="pca", random_state=42, perplexity=min(30, max(5, (len(X)//50))))
        if version.parse(sklearn.__version__) >= version.parse("1.2"): kws["max_iter"]=1000
        else: kws["n_iter"]=1000
        Z = TSNE(**kws).fit_transform(Xp)

    plt.figure(figsize=(6,5))
    plt.scatter(Z[y==0,0], Z[y==0,1], s=6, alpha=0.6, label="Domain A")
    plt.scatter(Z[y==1,0], Z[y==1,1], s=6, alpha=0.6, label="Domain B")
    plt.legend(); plt.title(f"{method.upper()} of Inception Features"); plt.tight_layout()

    out_png = Path(out_png)
    if name_suffix:
        out_png = out_png.with_name(out_png.stem + f"_{slugify(name_suffix)}" + out_png.suffix)
    plt.savefig(out_png, dpi=160); plt.close()
    return out_png
