import torch
import torch.nn as nn
from torchvision import transforms
from pathlib import Path

class DummySegModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        # Return random logits (B, C, H, W) for testing
        b, _, h, w = x.shape
        return torch.randn(b, self.num_classes, h, w, device=x.device)

def load_model(device):
    num_classes = 5  # match your --num_classes
    model = DummySegModel(num_classes=num_classes).to(device)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    model.eval()
    return model, transform
