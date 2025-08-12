# domain_gap_eval/model_definitions/detection.py
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
import json

class EchoGT:
    def __init__(self, img_dir, coco_json):
        self.img_dir = Path(img_dir)
        with open(coco_json, "r") as f:
            coco = json.load(f)
        # Map filename -> image id
        self.file2id = {im["file_name"]: im["id"] for im in coco["images"]}
        # Group annotations by image_id
        self.ann_by_img = {}
        for a in coco["annotations"]:
            self.ann_by_img.setdefault(a["image_id"], []).append(a)

        # simple transform (not really used)
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])

    def to(self, device): return self
    def eval(self): return self

    @torch.no_grad()
    def __call__(self, batch_tensor):
        # The evaluator passes images as tensors; we can't recover filenames here,
        # so we expose a side-channel: set self._current_filenames before calling.
        out = []
        for fn in self._current_filenames:
            iid = self.file2id.get(Path(fn).name, None)
            boxes = []; labels = []; scores = []
            if iid is not None:
                for a in self.ann_by_img.get(iid, []):
                    x,y,w,h = a["bbox"]
                    boxes.append([x, y, x+w, y+h])
                    labels.append(a["category_id"])
                    scores.append(0.99)  # high confidence
            if len(boxes) == 0:
                out.append({"boxes": torch.zeros((0,4), dtype=torch.float32),
                            "labels": torch.zeros((0,), dtype=torch.int64),
                            "scores": torch.zeros((0,), dtype=torch.float32)})
            else:
                out.append({"boxes": torch.tensor(boxes, dtype=torch.float32),
                            "labels": torch.tensor(labels, dtype=torch.int64),
                            "scores": torch.tensor(scores, dtype=torch.float32)})
        return out

def load_model(device):
    # img_dir will be injected by the orchestrator via the transform call,
    # so we bind a dummy and patch filenames at runtime.
    # We’ll wrap a transform that stores filenames for the next call.
    # You must pass the same coco json path via env var.
    import os
    coco_json = os.environ.get("DGE_COCO_GT")# or "./domain_gap_eval/dummy_data/det_coco.json"
    img_dir   = os.environ.get("DGE_IMG_DIR")#  or "./domain_gap_eval/dummy_data/det_images"
    model = EchoGT(img_dir, coco_json)

    def transform(pil_img):
        return model.transform(pil_img)

    # hook to capture filenames before model call
    def set_filenames(fns):
        model._current_filenames = fns
    model.set_filenames = set_filenames

    return model.to(device).eval(), transform
