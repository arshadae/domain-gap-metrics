from pathlib import Path
import json, numpy as np, torch
from config import seed_all, slugify
from notes import NOTES
from features.inception import InceptionFeat, featurize_paths
from metrics.fid_kid import compute_fid, compute_kid
from metrics.pad_mmd_cka import compute_pad, compute_mmd_rbf, compute_linear_cka
from metrics.lpips_dist import compute_lpips_set_distance
from metrics.hist import chi_square_rgb
from viz.embeddings import save_embedding_plot
from tasks.loader import load_user_model
from tasks.classification import evaluate_classification
from tasks.segmentation import evaluate_segmentation
from tasks.detection import evaluate_detection_coco

def run(args):
    seed_all(42)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # list images
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    def list_images(folder, max_images=None):
        paths = [p for p in sorted(Path(folder).rglob("*")) if p.suffix.lower() in IMG_EXTS]
        return paths[:max_images] if max_images else paths

    paths_a = list_images(args.domain_a, args.max_images)
    paths_b = list_images(args.domain_b, args.max_images)
    if not paths_a or not paths_b:
        raise RuntimeError("No images found. Check --domain_a/--domain_b and extensions.")

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    # features
    enc = InceptionFeat().to(device)
    feats_a = featurize_paths(paths_a, enc, batch_size=args.batch_size)
    feats_b = featurize_paths(paths_b, enc, batch_size=args.batch_size)

    # metrics
    muA, muB = feats_a.mean(0), feats_b.mean(0)
    sigA = np.cov(feats_a, rowvar=False); sigB = np.cov(feats_b, rowvar=False)
    fid = compute_fid(muA, sigA, muB, sigB)
    kid_mean, kid_std = compute_kid(feats_a, feats_b)
    pad, dom_acc = compute_pad(feats_a, feats_b)
    mmd2 = compute_mmd_rbf(feats_a, feats_b)
    
    
    # Clip negatives to zero
    fid = max(0.0, fid)
    kid_mean = max(0.0, kid_mean)
    mmd2 = max(0.0, mmd2)
    pad = max(0.0, pad)

    # Optional: symmetric PAD from domain classifier accuracy
    pad_symmetric = max(0.0, 2*(2*dom_acc - 1))
    
    cka = compute_linear_cka(feats_a, feats_b)
    lpips_val = compute_lpips_set_distance(paths_a, paths_b, sample_pairs=args.lpips_pairs, device=device) if args.lpips else None
    chi_rgb = chi_square_rgb(paths_a, paths_b)

    tsne_file = save_embedding_plot(feats_a, feats_b, out / "tsne.png", method="tsne", name_suffix=args.eval_name)
    umap_file = save_embedding_plot(feats_a, feats_b, out / "umap.png", method="umap", name_suffix=args.eval_name) if True else None

    # optional task metrics
    task_metrics = None
    if args.task:
        if not args.model_py:
            raise ValueError("--model_py is required when --task is set.")
        model, tform = load_user_model(args.model_py, device)
        if args.task == "classification":
            if not args.labels_csv: raise ValueError("--labels_csv required for classification")
            task_metrics = evaluate_classification(args.domain_a, args.labels_csv, model, tform, device, args.batch_size)
        elif args.task == "segmentation":
            if (args.mask_dir is None) or (args.num_classes is None):
                raise ValueError("--mask_dir and --num_classes required for segmentation")
            task_metrics = evaluate_segmentation(args.domain_a, args.mask_dir, model, tform, device,
                                                 num_classes=args.num_classes, ignore_index=args.ignore_index,
                                                 mask_suffix=args.mask_suffix, batch_size=args.batch_size)
        else:
            if not args.coco_gt: raise ValueError("--coco_gt required for detection")
            task_metrics = evaluate_detection_coco(args.domain_a, args.coco_gt, model, tform, device,
                                                   score_thresh=args.score_thresh, batch_size=1)

    summary = {
        "evaluation_name": args.eval_name,
        "num_images": {"A": len(paths_a), "B": len(paths_b)},
        "features": "InceptionV3 pool3 (2048-D)",
        "metrics": {
            "FID": fid, "KID_mean": kid_mean, "KID_std": kid_std,
            "PAD": pad, "PAD_symmetric": pad_symmetric, "DomainClassifierAccuracy": dom_acc,
            "MMD2_RBF": mmd2, "CKA_linear_paired": cka,
            "LPIPS_set_to_set": lpips_val, "ChiSquare_RGB_Hist": chi_rgb
        },
        "plots": {"TSNE": str(tsne_file), "UMAP": str(umap_file)},
        "task_metrics": task_metrics,
        "notes": NOTES
    }
    json_name = f"summary_{slugify(args.eval_name)}.json" if args.eval_name else "summary.json"
    with open(out / json_name, "w") as f: json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
