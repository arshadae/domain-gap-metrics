from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch

def pil_loader(path):
    with Image.open(path) as im: return im.convert("RGB")

def patch_numpy_aliases():
    import numpy as _np
    for name, repl in [("float", float), ("int", int), ("bool", bool), ("object", object), ("long", int)]:
        if not hasattr(_np, name): setattr(_np, name, repl)

def evaluate_detection_coco(img_dir, coco_gt_json, model, transform, device,
                            score_thresh=0.0, batch_size=1):
    patch_numpy_aliases()
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(coco_gt_json)
    img_ids = coco_gt.getImgIds()
    id2file = {i["id"]: i["file_name"] for i in coco_gt.loadImgs(img_ids)}
    valid_ids = [i for i in img_ids if (Path(img_dir) / id2file[i]).exists()]

    dets = []
    for idx in tqdm(range(0, len(valid_ids), batch_size), desc="Det inference", leave=False):
        bid = valid_ids[idx:idx+batch_size]
        ims = [pil_loader(Path(img_dir) / id2file[i]) for i in bid]
        filenames = [id2file[i] for i in bid]
        x = torch.stack([transform(im) for im in ims]).to(device)
        with torch.no_grad():
            if hasattr(model, "set_filenames"):
                model.set_filenames(filenames)
            out = model(x)

        if isinstance(out, dict): out = [out]
        for i, oid in enumerate(bid):
            boxes = out[i]["boxes"].detach().cpu().numpy()
            scores = out[i]["scores"].detach().cpu().numpy()
            labels = out[i]["labels"].detach().cpu().numpy()
            xywh = boxes.copy(); xywh[:,2] -= xywh[:,0]; xywh[:,3] -= xywh[:,1]
            for j in range(xywh.shape[0]):
                if scores[j] < score_thresh: continue
                dets.append({
                    "image_id": int(oid),
                    "category_id": int(labels[j]),
                    "bbox": [float(xywh[j,0]), float(xywh[j,1]), float(xywh[j,2]), float(xywh[j,3])],
                    "score": float(scores[j])
                })
    if not dets: raise RuntimeError("No detections produced; check model and --score_thresh.")
    coco_dt = coco_gt.loadRes(dets)
    E = COCOeval(coco_gt, coco_dt, iouType="bbox")
    E.evaluate(); E.accumulate(); E.summarize()
    s = E.stats
    return {
        "task": "detection",
        "COCO_metrics": {
            "AP@[0.50:0.95]": float(s[0]), "AP50": float(s[1]), "AP75": float(s[2]),
            "AP_small": float(s[3]), "AP_medium": float(s[4]), "AP_large": float(s[5]),
            "AR_1": float(s[6]), "AR_10": float(s[7]), "AR_100": float(s[8]),
            "AR_small": float(s[9]), "AR_medium": float(s[10]), "AR_large": float(s[11]),
        },
        "num_images": int(len(valid_ids)),
        "score_thresh": float(score_thresh),
    }
