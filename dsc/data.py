"""The eight collections of the paper (and the DomainNet subset of the literature protocol).

Each loader returns (source, labels, domains): `source.get(i)` gives a PIL image, labels are class names and domains
are depiction names ("single" for single-depiction collections). Subsets are drawn with fixed seeds.
"""
import glob
import os
import re
import subprocess
import zipfile

import numpy as np
from PIL import Image

COREL1K_REPO = "https://github.com/zitansmail/CBIR-New-Search-Engine-Approach"
COREL10K_URL = "https://www.kaggle.com/api/v1/datasets/download/michelwilson/corel10k"
DN_PER_CELL = 6                   # DomainNet test split: at most 6 images per (class, domain)
IMG_EXT = ("jpg", "jpeg", "png", "bmp", "JPG", "JPEG", "PNG", "BMP")
MIXED = ["pacs", "officehome", "domainnet", "imagenetr"]
SINGLE = ["corel1k", "corel10k", "caltech101", "caltech256"]
NAMES = {"pacs": "PACS", "officehome": "Office-Home", "domainnet": "DomainNet", "imagenetr": "ImageNet-R",
         "corel1k": "Corel-1K", "corel10k": "Corel-10K", "caltech101": "Caltech-101", "caltech256": "Caltech-256",
         "domainnet7": "DomainNet (7 classes)"}


class PathSource:
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def get(self, i): return Image.open(self.paths[i]).convert("RGB")


class HFSource:
    def __init__(self, ds, idx, col="image"): self.ds, self.idx, self.col = ds, idx, col
    def __len__(self): return len(self.idx)
    def get(self, i): return self.ds[int(self.idx[i])][self.col].convert("RGB")


def cap_per_cell(labels, domains, cap, seed=0):
    """At most `cap` images per (class, domain), random order with a fixed seed; cap=None keeps everything."""
    order = np.random.RandomState(seed).permutation(len(labels)) if cap else np.arange(len(labels))
    keep, seen = [], {}
    for i in order:
        key = (labels[i], domains[i])
        if not cap or seen.get(key, 0) < cap:
            keep.append(i); seen[key] = seen.get(key, 0) + 1
    return np.sort(np.array(keep, dtype=int))


def _names(ds, col):
    v = np.array(ds[col])
    return np.array(ds.features[col].names)[v] if hasattr(ds.features[col], "names") else v.astype(str)


def _hf_mixed(hf_id, split, cap):
    from datasets import load_dataset
    ds = load_dataset(hf_id, split=split)
    lab, dom = _names(ds, "label"), _names(ds, "domain")
    idx = cap_per_cell(lab, dom, cap)
    return HFSource(ds, idx), lab[idx], dom[idx]


def load(name, root="data", cap=None):
    """cap: at most `cap` images per (class, domain) for quick tests (None = the paper's setting)."""
    if name == "pacs":
        return _hf_mixed("flwrlabs/pacs", "train", cap)
    if name == "officehome":
        return _hf_mixed("flwrlabs/office-home", "train", cap)
    if name == "domainnet":
        return _hf_mixed("wltjr1007/DomainNet", "test", cap or DN_PER_CELL)
    if name == "domainnet7":
        return _domainnet7(cap)
    if name == "imagenetr":
        return _imagenetr(cap)
    if name == "caltech101":
        from datasets import load_dataset
        ds = load_dataset("flwrlabs/caltech101", split="train")          # 101 classes, no background class
        lab = np.array(ds["label"]).astype(str)
        idx = cap_per_cell(lab, np.zeros(len(lab)), cap)
        return HFSource(ds, idx), lab[idx], np.array(["single"] * len(idx))
    if name == "caltech256":
        return _caltech256(cap)
    if name == "corel1k":
        return _corel1k(os.path.join(root, "corel1k"), cap)
    if name == "corel10k":
        return _corel10k(os.path.join(root, "corel10k"), cap)
    raise ValueError(name)


def _domainnet7(cap):
    """Classes with more than 200 images in every domain (Hu & Lee, ECCV 2022), over train + test."""
    from datasets import load_dataset, concatenate_datasets
    import pandas as pd
    d = {s: load_dataset("wltjr1007/DomainNet", split=s) for s in ("train", "test")}
    ds = concatenate_datasets([d["train"], d["test"]])
    lab, dom = _names(ds, "label"), _names(ds, "domain")
    counts = pd.crosstab(lab, dom)
    keep = list(counts.index[(counts > 200).all(1)])
    assert len(keep) == 7, f"expected 7 classes, got {len(keep)}"
    idx = np.where(np.isin(lab, keep))[0]
    idx = idx[cap_per_cell(lab[idx], dom[idx], cap)]
    return HFSource(ds, idx), lab[idx], dom[idx]


def _imagenetr(cap):
    """Keys look like 'imagenet-rendition/imagenet-r/n02480495/sketch_14': class = wnid, rendition before '_<n>'."""
    from datasets import load_dataset
    ds = load_dataset("koorye/ImageNet-Rendition", split="train")
    parts = [k.split("/") for k in ds["__key__"]]
    ok = np.array([len(p) >= 2 and p[-2].startswith("n") and "_" in p[-1] for p in parts])
    lab = np.array([p[-2] if o else "" for p, o in zip(parts, ok)])
    dom = np.array([p[-1].rsplit("_", 1)[0] if o else "" for p, o in zip(parts, ok)])
    idx = np.where(ok)[0]
    idx = idx[cap_per_cell(lab[idx], dom[idx], cap)]
    return HFSource(ds, idx, col="jpg"), lab[idx], dom[idx]


def _caltech256(cap):
    from datasets import load_dataset, concatenate_datasets
    d = load_dataset("ilee0022/Caltech-256")
    ds = concatenate_datasets([d[s] for s in sorted(d)])
    text = np.array([str(t) for t in ds["text"]])
    clutter = (np.char.find(np.char.lower(text), "clutter") >= 0) | (np.array(ds["label"]) == 257)
    keep = np.where(~clutter)[0]
    keep = keep[cap_per_cell(text[keep], np.zeros(len(keep)), cap)]
    if not cap:
        assert len(set(text[keep])) == 256, "Caltech-256: 256 classes expected"
    return HFSource(ds, keep), text[keep], np.array(["single"] * len(keep))


def _corel1k(root, cap):
    if not os.path.isdir(root):
        subprocess.run(["git", "clone", "-q", COREL1K_REPO, root], check=True)
    base = os.path.join(root, "dataset")
    paths, labels = [], []
    for split in ("training_set", "test_set"):
        for cls in sorted(os.listdir(os.path.join(base, split))):
            files = sorted(glob.glob(os.path.join(base, split, cls, "*.jpg")))
            paths += files; labels += [cls] * len(files)
    labels = np.array(labels)
    idx = cap_per_cell(labels, np.zeros(len(labels)), cap)
    return PathSource([paths[i] for i in idx]), labels[idx], np.array(["single"] * len(idx))


def _corel10k_dir(root):
    if not os.path.isdir(root):
        os.makedirs(root, exist_ok=True)
        z = os.path.join(root, "corel10k.zip")
        subprocess.run(["curl", "-sL", "-o", z, COREL10K_URL], check=True)
        zipfile.ZipFile(z).extractall(root)
        os.remove(z)
    entries = [e for e in os.listdir(root) if not e.startswith((".", "__"))]
    if len(entries) == 1 and os.path.isdir(os.path.join(root, entries[0])):
        return _corel10k_dir(os.path.join(root, entries[0]))          # Kaggle archive: one folder "Corek-10k"
    return root


def _corel10k(root, cap):
    """Kaggle copy: 0.jpeg ... 9999.jpeg, class = n // 100 (checked visually at boundaries 100, 200, 5000)."""
    root = _corel10k_dir(root)
    subdirs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if subdirs:
        paths, labels = [], []
        for cls in subdirs:
            files = sorted(f for e in IMG_EXT for f in glob.glob(os.path.join(root, cls, f"*.{e}")))
            paths += files; labels += [cls] * len(files)
    else:
        num = lambda f: int(re.findall(r"\d+", os.path.basename(f))[0])
        paths = sorted((f for e in IMG_EXT for f in glob.glob(os.path.join(root, f"*.{e}"))), key=num)
        first = num(paths[0])
        labels = [str((num(f) - first) // 100) for f in paths]
    labels = np.array(labels)
    assert len(paths) == 10000 and len(set(labels)) == 100, "Corel-10K incomplete"
    idx = cap_per_cell(labels, np.zeros(len(labels)), cap)
    return PathSource([paths[i] for i in idx]), labels[idx], np.array(["single"] * len(idx))
