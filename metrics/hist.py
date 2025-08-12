import numpy as np
from PIL import Image

def chi_square_rgb(paths_a, paths_b, bins=64):
    def mean_hist(paths):
        h = np.zeros((3,bins), dtype=np.float64)
        for p in paths:
            arr = np.asarray(Image.open(p).convert("RGB").resize((256,256)), dtype=np.uint8)
            for c in range(3):
                hist,_ = np.histogram(arr[...,c], bins=bins, range=(0,255))
                h[c]+=hist
        h += 1e-8; h = h / h.sum(axis=1, keepdims=True)
        return h
    A,B = mean_hist(paths_a), mean_hist(paths_b)
    return float(0.5 * np.sum((A-B)**2 / (A+B)))
