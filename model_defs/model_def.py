# model_def.py
# This file is provided as a reference for a general model definition.
# It is not used in the actual implementation of the segmentation model, classification model, and Detection model.

# model_cls.py
import torch
from torchvision import models, transforms
from pathlib import Path

def load_model(device):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features,  num_classes := 10)  # adjust
    ckpt = "path/to/your_cls_ckpt.pth"  # optional
    if ckpt and Path(ckpt).exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    tf = models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    return model, tf




# model_seg.py
import torch
from torchvision import models
from torchvision.transforms import functional as TF
from pathlib import Path

def _seg_tf(im):
    im = TF.resize(im, [512, 512])
    t = TF.to_tensor(im)
    t = TF.normalize(t, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    return t

def load_model(device):
    model = models.segmentation.deeplabv3_resnet50(weights="DEFAULT")
    model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels,  num_classes := 7, 1)  # adjust
    ckpt = "path/to/your_seg_ckpt.pth"
    if ckpt and Path(ckpt).exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model, _seg_tf




# model_det.py (COCO Style)
import torch
from torchvision import models
from torchvision.transforms import functional as TF
from pathlib import Path

def _det_tf(im):
    im = TF.resize(im, [800])  # keep aspect
    t = TF.to_tensor(im)
    t = TF.normalize(t, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    return t

def load_model(device):
    model = models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # Optionally replace head for your label set:
    num_classes = 91  # adjust to your dataset (incl. background=0)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    ckpt = "path/to/your_det_ckpt.pth"
    if ckpt and Path(ckpt).exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model, _det_tf

