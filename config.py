import re, random, numpy as np, torch

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")[:120]
