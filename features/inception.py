import torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def pil_loader(path):
    with Image.open(path) as im:
        return im.convert("RGB")

class InceptionFeat(nn.Module):
    """InceptionV3 pool3/logits (2048-D) features, version-robust."""
    def __init__(self):
        super().__init__()
        try:
            weights = models.Inception_V3_Weights.IMAGENET1K_V1
            net = models.inception_v3(weights=weights, aux_logits=True)
            self.tf = weights.transforms()
        except Exception:
            net = models.inception_v3(weights="IMAGENET1K_V1", aux_logits=True)
            self.tf = transforms.Compose([
                transforms.Resize(342), transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])
        net.fc = nn.Identity()
        net.eval()
        for p in net.parameters(): p.requires_grad = False
        self.backbone = net

    @property
    def device(self):
        return next(self.backbone.parameters()).device

    @torch.no_grad()
    def forward(self, pil_list):
        x = torch.stack([self.tf(im) for im in pil_list], 0).to(self.device)
        out = self.backbone(x)
        if hasattr(out, "logits"): out = out.logits
        elif isinstance(out, (tuple, list)): out = out[0]
        return out  # (N, 2048)

@torch.no_grad()
def featurize_paths(paths, extractor, batch_size=32):
    feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="Featurizing", leave=False):
        batch = [pil_loader(p) for p in paths[i:i+batch_size]]
        f = extractor(batch).cpu().numpy()
        feats.append(f)
    if not feats: return np.zeros((0, 2048), dtype=np.float32)
    return np.concatenate(feats, 0)
