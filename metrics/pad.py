# pad_minimal.py
# Proxy A-distance (PAD) per Ben-David et al. using a linear domain classifier.
# Requires: numpy, scikit-learn

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

Classifier = Literal["logreg", "linsvm"]
Norm = Literal["zscore", "l2", "none"]

@dataclass
class PADResult:
    pad_mean: float
    pad_std: float
    acc_mean: float
    acc_std: float
    splits: int
    clf: str
    norm: str
    C: float
    balanced: bool

def pad_from_accuracy(acc: float) -> float:
    """
    Map accuracy to PAD with symmetry (flip if < 0.5).
    PAD = 4 * max(acc, 1-acc) - 2  in [0, 2].
    """
    acc_eff = acc if acc >= 0.5 else (1.0 - acc)
    return 4.0 * acc_eff - 2.0

def _standardize(X_tr, X_te, norm: Norm):
    if norm == "zscore":
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
    elif norm == "l2":
        X_tr = normalize(X_tr, norm="l2")
        X_te = normalize(X_te, norm="l2")
    elif norm == "none":
        pass
    else:
        raise ValueError("norm must be 'zscore', 'l2', or 'none'")
    return X_tr, X_te

def _build_clf(clf: Classifier, C: float):
    """
    Use linear classifiers so the hypothesis class is fixed and simple.
    """
    if clf == "logreg":
        return LogisticRegression(
            penalty="l2", C=C, solver="liblinear",
            class_weight="balanced", max_iter=1000
        )
    elif clf == "linsvm":
        return LinearSVC(C=C, class_weight="balanced")
    else:
        raise ValueError("clf must be 'logreg' or 'linsvm'")

def compute_pad(
    X_a: np.ndarray,
    X_b: np.ndarray,
    *,
    clf: Classifier = "logreg",
    C: float = 1.0,
    norm: Norm = "zscore",
    test_size: float = 0.2,
    n_splits: int = 10,
    random_state: int = 0,
    balance_classes: bool = True,
) -> PADResult:
    """
    Compute Proxy A-distance (PAD) between distributions A and B given feature arrays.

    Parameters
    ----------
    X_a, X_b : arrays of shape (Na, D) and (Nb, D)
        Frozen feature embeddings (e.g., Inception pool3, task model penultimate).
    clf : {'logreg','linsvm'}
        Linear domain classifier family.
    C : float
        Regularization strength.
    norm : {'zscore','l2','none'}
        Feature normalization (zscore recommended).
    test_size : float
        Fraction for held-out test set in each random split.
    n_splits : int
        Number of random stratified splits.
    balance_classes : bool
        If True, subsample the larger domain so A and B have equal counts.

    Returns
    -------
    PADResult with mean±std of PAD and accuracy across splits.

    Notes
    -----
    - Per Ben-David et al., PAD uses the empirical error of a domain classifier.
    - We use linear models trained with convex surrogates (logistic/hinge),
      which is the standard practical estimator.
    """
    rng = np.random.RandomState(random_state)

    # Optional: balance A/B counts to avoid trivial imbalance effects
    if balance_classes:
        n = min(len(X_a), len(X_b))
        idx_a = rng.choice(len(X_a), size=n, replace=False)
        idx_b = rng.choice(len(X_b), size=n, replace=False)
        X_a = X_a[idx_a]
        X_b = X_b[idx_b]

    X = np.concatenate([X_a, X_b], axis=0)
    y = np.concatenate([np.zeros(len(X_a), dtype=int),
                        np.ones(len(X_b), dtype=int)])

    splitter = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=random_state
    )

    pads, accs = [], []

    for tr_idx, te_idx in splitter.split(X, y):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        X_tr, X_te = _standardize(X_tr, X_te, norm)
        est = _build_clf(clf, C)
        est.fit(X_tr, y_tr)

        y_pred = est.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        pad = pad_from_accuracy(acc)

        accs.append(acc)
        pads.append(pad)

    return PADResult(
        pad_mean=float(np.mean(pads)),
        pad_std=float(np.std(pads, ddof=1)) if n_splits > 1 else 0.0,
        acc_mean=float(np.mean(accs)),
        acc_std=float(np.std(accs, ddof=1)) if n_splits > 1 else 0.0,
        splits=n_splits,
        clf=clf,
        norm=norm,
        C=C,
        balanced=balance_classes,
    )

# ----- Example usage -----
if __name__ == "__main__":
    # Toy demo with synthetic features:
    rng = np.random.RandomState(42)
    Xa = rng.normal(0, 1, size=(1000, 128))
    Xb = rng.normal(0.2, 1, size=(1000, 128))  # small shift

    res = compute_pad(Xa, Xb, clf="logreg", C=1.0, norm="zscore",
                      n_splits=10, test_size=0.2, random_state=0)

    print(f"PAD: {res.pad_mean:.3f} ± {res.pad_std:.3f}  (0=indist., 2=max)")
    print(f"Acc: {res.acc_mean:.3f} ± {res.acc_std:.3f}")
