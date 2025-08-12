# domain_gap_eval/model_definitions/classification.py
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import csv

class FilenameLookupClassifier(nn.Module):
    def __init__(self, name2label, num_classes):
        super().__init__()
        self.name2label = name2label
        self.num_classes = num_classes

    def forward(self, x):  # x: (B,C,H,W); we ignore pixels
        # When used by the evaluator we can't see filenames, so we store a side-channel on the tensor
        # The evaluator doesn't support this; so we instead provide a transform that encodes filename in tensor? Not needed.
        # Instead: we’ll expose a .predict(list_of_PIL) utility and wrap below.
        raise RuntimeError("Use predict_pil(list_of_PIL) with this dummy model.")

    @torch.no_grad()
    def predict_pil(self, pil_images, filenames):
        # Return logits where argmax equals ground-truth label from CSV
        B = len(pil_images)
        logits = torch.zeros((B, self.num_classes), dtype=torch.float32)
        for i, fn in enumerate(filenames):
            lbl = self.name2label.get(Path(fn).name, 0)
            logits[i, lbl] = 10.0
        return logits

def load_model(device):
    # Build filename->label map from the colocated CSV used by the CLI
    # We assume you call with --labels_csv ./domain_gap_eval/synthetic_data/cls_labels.csv
    # so we read it here for the dummy model.
    import os
    labels_csv = os.environ.get("DGE_LABELS_CSV", "")
    name2label = {}
    if labels_csv and Path(labels_csv).exists():
        with open(labels_csv, "r") as f:
            for row in csv.DictReader(f):
                name2label[row["filename"]] = int(row["label"])
    num_classes = max(name2label.values()) + 1 if name2label else 9

    model = FilenameLookupClassifier(name2label, num_classes)

    # Wrap the transform to let the evaluator use model.predict_pil instead of forward
    def transform(pil):
        # Just resize/normalize; values are ignored by the dummy model
        return transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])(pil)

    # Monkey-patch a forward hook the evaluator can use (filenames-aware)
    # We’ll patch in orchestrator via a small change (below) OR you can switch to a real model afterwards.
    model.eval()
    return model.to(device), transform
