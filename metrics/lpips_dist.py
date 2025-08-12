def compute_lpips_set_distance(paths_a, paths_b, sample_pairs=500, device="cuda"):
    try:
        import lpips, torch
        from torchvision import transforms
        from PIL import Image
        import random, numpy as np
    except Exception:
        return None
    loss_fn = lpips.LPIPS(net='alex').to(device).eval()
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    nA, nB = len(paths_a), len(paths_b)
    if nA == 0 or nB == 0: return None
    sp = min(sample_pairs, nA*nB)
    vals=[]
    with torch.no_grad():
        for _ in range(sp):
            ia, ib = random.randrange(nA), random.randrange(nB)
            a = tf(Image.open(paths_a[ia]).convert("RGB")).unsqueeze(0).to(device)
            b = tf(Image.open(paths_b[ib]).convert("RGB")).unsqueeze(0).to(device)
            vals.append(loss_fn(a*2-1, b*2-1).item())
    return float(sum(vals)/len(vals))
