"""Frozen descriptors of the paper, extracted once and cached.

DINOv2 ViT-B/14 and CLIP ViT-B/16 (timm): L2([L2(class token); L2(mean of patch tokens)]), short side resized to 256,
centre crop 224. SigLIP 2 base: its image embedding with its own processor. Fusion: equal-weight concatenation.
"""
import os

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .method import SIGLIP_ID, as_tensor, load_siglip

TIMM_IDS = {"dinov2": "vit_base_patch14_dinov2.lvd142m", "clip": "vit_base_patch16_clip_224.openai"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = ["DINOv2", "CLIP", "SigLIP 2", "Fusion"]


def _amp():
    return torch.autocast("cuda", dtype=torch.float16) if DEVICE == "cuda" else torch.autocast("cpu", enabled=False)


@torch.no_grad()
def vit_descriptor(source, name, bs=64):
    import timm
    from torchvision import transforms as T
    bb = timm.create_model(TIMM_IDS[name], pretrained=True, num_classes=0, img_size=224).to(DEVICE).eval()
    tf = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize(bb.pretrained_cfg["mean"], bb.pretrained_cfg["std"])])
    out = []
    for s in tqdm(range(0, len(source), bs), desc=name, leave=False):
        x = torch.stack([tf(source.get(i)) for i in range(s, min(s + bs, len(source)))]).to(DEVICE)
        with _amp():
            f = bb.forward_features(x).float()
        c, m = f[:, 0], f[:, bb.num_prefix_tokens:].mean(1)
        out.append(F.normalize(torch.cat([F.normalize(c, dim=1), F.normalize(m, dim=1)], 1), dim=1).cpu())
    return torch.cat(out)


@torch.no_grad()
def siglip_embeddings(source, bs=64, model_id=SIGLIP_ID):
    m, p = load_siglip(model_id, DEVICE)
    out = []
    for s in tqdm(range(0, len(source), bs), desc="siglip2", leave=False):
        x = p(images=[source.get(i) for i in range(s, min(s + bs, len(source)))], return_tensors="pt").to(DEVICE)
        out.append(F.normalize(as_tensor(m.get_image_features(**x)).float(), dim=-1).cpu())
    return torch.cat(out)


@torch.no_grad()
def style_statistics(source, blocks=(1, 2, 3), bs=64):
    """Per-channel mean and std of the patch tokens of early DINOv2 blocks (k-means ablation of the paper)."""
    import timm
    from torchvision import transforms as T
    bb = timm.create_model(TIMM_IDS["dinov2"], pretrained=True, num_classes=0, img_size=224).to(DEVICE).eval()
    tf = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize(bb.pretrained_cfg["mean"], bb.pretrained_cfg["std"])])
    out = []
    for s in tqdm(range(0, len(source), bs), desc="style", leave=False):
        x = torch.stack([tf(source.get(i)) for i in range(s, min(s + bs, len(source)))]).to(DEVICE)
        with _amp():
            hs = bb.forward_intermediates(x, indices=list(blocks), output_fmt="NLC", intermediates_only=True,
                                          stop_early=True)
        out.append(torch.cat([F.normalize(torch.cat([h.float().mean(1), h.float().std(1)], 1), dim=1)
                              for h in hs], 1).cpu())
    return torch.cat(out)


def cached(path, fn):
    if os.path.exists(path):
        return torch.load(path)
    x = fn()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(x, path)
    return x


def descriptors(name, source, cache="cache", tag=""):
    """The four descriptors of a collection: DINOv2, CLIP, SigLIP 2 and their fusion."""
    X = {"DINOv2": cached(f"{cache}/{name}{tag}_dinov2.pt", lambda: vit_descriptor(source, "dinov2")),
         "CLIP": cached(f"{cache}/{name}{tag}_clip.pt", lambda: vit_descriptor(source, "clip")),
         "SigLIP 2": cached(f"{cache}/{name}{tag}_siglip2.pt", lambda: siglip_embeddings(source))}
    X["Fusion"] = torch.cat([F.normalize(v, dim=1) for v in X.values()], 1)
    return X
