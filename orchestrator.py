# orchestrator.py
from pathlib import Path
import json, numpy as np, torch

from config import seed_all, slugify
from notes import NOTES

from pytorch_fid import fid_score, inception
from metrics.mmd2_poly import kid_from_features_mmdgan
from metrics.mmd2_rbf import mmd2_rbf_from_features
from metrics.lpips_dist import compute_lpips_set_distance
from metrics.pad import compute_pad
from metrics.ssim import compute_ssim_dirs


from viz.embeddings import save_embedding_plot
from tasks.loader import load_user_model

from tasks.classification import evaluate_classification
from tasks.segmentation import evaluate_segmentation
from tasks.detection import evaluate_detection_coco


def run(args):
    seed_all(42)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    # InceptionV3 pool3 (2048-D) for FID/KID/MMD features
    dims = 2048
    block_idx = inception.InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = inception.InceptionV3([block_idx]).to(device)

    # -------- image listing --------
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    def list_images(folder, max_images=None):
        paths = [p for p in sorted(Path(folder).rglob("*"))
                 if p.suffix.lower() in IMG_EXTS]
        return paths[:max_images] if max_images else paths

    paths_a = list_images(args.domain_a, args.max_images)
    paths_b = list_images(args.domain_b, args.max_images)
    if not paths_a or not paths_b:
        raise RuntimeError("No images found. Check --domain_a/--domain_b and extensions.")

    print('Running domain gap evaluations on device:', device)
    print(f"Domain A: {len(paths_a)} images from {args.domain_a}")
    print(f"Domain B: {len(paths_b)} images from {args.domain_b}")

    # -------- features & stats --------
    print("Extracting features for domain A...")
    feats_a = fid_score.get_activations(paths_a, model, args.batch_size, dims, device, 1)
    muA, sigA = np.mean(feats_a, axis=0), np.cov(feats_a, rowvar=False)

    print("Extracting features for domain B...")
    feats_b = fid_score.get_activations(paths_b, model, args.batch_size, dims, device, 1)
    muB, sigB = np.mean(feats_b, axis=0), np.cov(feats_b, rowvar=False)
    print("Feature extraction done.")

    # -------- FID --------
    print("Computing FID ...")
    fid = fid_score.calculate_frechet_distance(muA, sigA, muB, sigB)

    # -------- KID (poly/MMD-GAN style) --------
    print("Computing KID (MMD2 with polynomial kernel) ...")
    kid_mean, kid_std = kid_from_features_mmdgan(
        feats_a, feats_b,
        subsets=100, subset_size=1000,
        seed=123, device=None, dtype=None
    )

    # -------- MMD2 (RBF) --------
    print("Computing MMD2 with RBF kernel (multi-sigma) ...")
    mmd2, mmd2_std = mmd2_rbf_from_features(
        feats_a, feats_b,
        sigmas="auto-median-x{0.5,1,2,4,8}",
        subsets=100, subset_size=1000,
        seed=123, device=None, dtype=torch.float64
    )
    print("MMD2_RBF:", mmd2, "(±" + f"{mmd2_std:.6g}" + ")" if mmd2_std is not None else "")

    print("Computing MMD2 with fixed sigmas ...")
    mmd2_fixed_sigma, _ = mmd2_rbf_from_features(
        feats_a, feats_b,
        sigmas=[10.0, 20.0, 40.0, 80.0],
        subsets=1, subset_size=None
    )

    print("Computing MMD2 with single auto-median sigma ...")
    mmd2_single_bw, _ = mmd2_rbf_from_features(
        feats_a, feats_b,
        sigmas="auto-median",
        subsets=1, subset_size=None
    )

    # -------- LPIPS (set-to-set) --------
    print("Computing LPIPS set-to-set distance...")
    lpips_val = compute_lpips_set_distance(paths_a, paths_b,
                                           sample_pairs=args.lpips_pairs,
                                           device=device) if getattr(args, "lpips", True) else None

    # -------- PAD (Proxy A-distance) --------
    # Paper-faithful PAD via linear domain classifier on frozen features
    print("Computing Proxy A-distance (PAD) on Inception features...")
    pad_res = compute_pad(
        feats_a, feats_b,
        clf="logreg",        # or "linsvm"
        C=1.0,
        norm="zscore",
        test_size=0.2,
        n_splits=10,
        random_state=0,
        balance_classes=True
    )
    pad, pad_std = pad_res.pad_mean, pad_res.pad_std
    dom_acc, dom_acc_std = pad_res.acc_mean, pad_res.acc_std
    pad_symmetric = pad  # mapping already enforces symmetry
    print(f"PAD: {pad:.6f} ± {pad_std:.6f}  |  Acc: {dom_acc:.4f} ± {dom_acc_std:.4f}")

    # -------- SSIM (Structural Similarity Index Matrix) --------
    print("Computing SSIM on paired images...")
    ssim_pairing = getattr(args, "ssim_pairing", "name")   # 'name'|'stem'|'sorted'
    ssim_mode    = getattr(args, "ssim_mode", "y")         # 'y'|'gray'|'rgb'
    ssim_resize  = not getattr(args, "ssim_no_resize", False)
    ssim_csv     = getattr(args, "ssim_csv", None)

    ssim_res = compute_ssim_dirs(
        Path(args.domain_a), Path(args.domain_b),
        pairing=ssim_pairing, mode=ssim_mode,
        resize_b_to_a=ssim_resize, save_csv=ssim_csv
    )
    print(f"SSIM ({ssim_res['mode']} / {ssim_res['pairing']}): "
        f"{ssim_res['mean']:.6f} ± {ssim_res['std']:.6f} over {ssim_res['pairs']} pairs")



    # -------- Embedding plots --------
    print("Saving embedding plots (this may take a while for large sets)...")
    tsne_file = save_embedding_plot(feats_a, feats_b, out / "tsne.png",
                                    method="tsne", name_suffix=args.eval_name)
    umap_file = save_embedding_plot(feats_a, feats_b, out / "umap.png",
                                    method="umap", name_suffix=args.eval_name)
    print(f"TSNE plot saved to {tsne_file}")
    if umap_file: print(f"UMAP plot saved to {umap_file}")

    # -------- Optional downstream task metrics --------
    task_metrics = None
    if getattr(args, "task", None):
        if not args.model_py:
            raise ValueError("--model_py is required when --task is set.")
        model, tform = load_user_model(args.model_py, device)
        if args.task == "classification":
            if not args.labels_csv:
                raise ValueError("--labels_csv required for classification")
            task_metrics = evaluate_classification(
                args.domain_a, args.labels_csv, model, tform, device, args.batch_size)
        elif args.task == "segmentation":
            if (args.mask_dir is None) or (args.num_classes is None):
                raise ValueError("--mask_dir and --num_classes required for segmentation")
            task_metrics = evaluate_segmentation(
                args.domain_a, args.mask_dir, model, tform, device,
                num_classes=args.num_classes, ignore_index=args.ignore_index,
                mask_suffix=args.mask_suffix, batch_size=args.batch_size)
        else:
            if not args.coco_gt:
                raise ValueError("--coco_gt required for detection")
            task_metrics = evaluate_detection_coco(
                args.domain_a, args.coco_gt, model, tform, device,
                score_thresh=args.score_thresh, batch_size=1)

    # -------- Summary --------
    summary = {
        "evaluation_name": args.eval_name,
        "num_images": {"A": len(paths_a), "B": len(paths_b)},
        "features": "InceptionV3 pool3 (2048-D)",
        "metrics": {
            "FID": float(fid),
            "KID_mean": float(kid_mean), "KID_std": float(kid_std),
            "MMD2_RBF": float(mmd2), "MMD2_RBF_std": (float(mmd2_std) if mmd2_std is not None else None),
            "MMD2_RBF_fixed_sigmas": float(mmd2_fixed_sigma),
            "MMD2_RBF_single_sigma": float(mmd2_single_bw),
            "LPIPS_set_to_set": (float(lpips_val) if lpips_val is not None else None),
            "PAD": float(pad), "PAD_std": float(pad_std),
            "PAD_symmetric": float(pad_symmetric),
            "DomainClassifierAccuracy": float(dom_acc),
            "DomainClassifierAccuracy_std": float(dom_acc_std),
            "SSIM_mean": float(ssim_res["mean"]),
            "SSIM_std": float(ssim_res["std"]),
            "SSIM_pairs": int(ssim_res["pairs"]),
            "SSIM_mode": ssim_res["mode"],
            "SSIM_pairing": ssim_res["pairing"],
        },
        "plots": {"TSNE": str(tsne_file), "UMAP": str(umap_file)},
        "task_metrics": task_metrics,
        "notes": NOTES,
    }

    json_name = f"summary_{slugify(args.eval_name)}.json" if args.eval_name else "summary.json"
    with open(out / json_name, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary written to {out / json_name}")
    print("Done.")
