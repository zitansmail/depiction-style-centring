"""Depiction Style Centring (DSC): training-free, label-free removal of depiction bias from frozen image embeddings.

1. a vision-language model (SigLIP 2) names the depiction of each image with a fixed list of generic words;
2. groups smaller than `min_group` images are merged into one group;
3. gate: if the largest group holds at least `tau` of the collection, the descriptors are returned unchanged;
4. otherwise each group is centred on its own mean.
The descriptor that is corrected can come from any frozen model (DINOv2, CLIP, SigLIP 2, a fusion...).
"""
import numpy as np
import torch
import torch.nn.functional as F

SIGLIP_ID = "google/siglip2-base-patch16-224"
# fixed before any result in the paper and used unchanged on every collection
DEPICTIONS = ["photo", "sketch", "pencil drawing", "line drawing", "doodle", "painting", "watercolor painting",
              "oil painting", "clipart", "cartoon", "infographic", "product photo on a white background", "3D render"]
TEMPLATES = ["a {}.", "a {} of an object.", "a {} of something.", "an image in the style of a {}."]
MIN_GROUP, TAU = 20, 2 / 3
MERGED = -1                       # group id of the merged small groups


def as_tensor(out):
    """transformers < 5 returns a tensor; recent versions return a ModelOutput whose pooler_output is the embedding."""
    if torch.is_tensor(out):
        return out
    pooled = getattr(out, "pooler_output", None)
    return pooled if pooled is not None else out[0]


def load_siglip(model_id=SIGLIP_ID, device="cpu"):
    from transformers import AutoModel, AutoProcessor
    return AutoModel.from_pretrained(model_id).to(device).eval(), AutoProcessor.from_pretrained(model_id)


@torch.no_grad()
def word_embeddings(model, processor, words=DEPICTIONS, templates=TEMPLATES, device="cpu"):
    """One L2-normalised text embedding per word: normalised mean of its templates (SigLIP 2 text encoder)."""
    out = []
    for w in words:
        t = processor(text=[tp.format(w) for tp in templates], padding="max_length", max_length=64,
                      return_tensors="pt").to(device)
        out.append(F.normalize(F.normalize(as_tensor(model.get_text_features(**t)).float(), dim=-1).mean(0), dim=0))
    return torch.stack(out).cpu()


def merge_small(groups, min_group=MIN_GROUP):
    g = np.asarray(groups).copy()
    ids, n = np.unique(g, return_counts=True)
    g[np.isin(g, ids[n < min_group])] = MERGED
    return g


def name_depictions(image_emb, word_emb, min_group=MIN_GROUP):
    """Index of the closest depiction word for each image (Eq. 1), small groups merged."""
    g = (F.normalize(image_emb.float(), dim=1) @ word_emb.float().T).argmax(1).numpy()
    return merge_small(g, min_group)


def gate_open(groups, tau=TAU):
    """DSC applies only if no group holds `tau` or more of the collection (Eq. 3)."""
    return np.unique(groups, return_counts=True)[1].max() / len(groups) < tau


def centre_by_group(x, groups):
    """z_i = x_i - mean of the descriptors of its group (Eq. 2), on L2-normalised descriptors."""
    x = F.normalize(x.float(), dim=1)
    g = torch.tensor(np.unique(groups, return_inverse=True)[1])
    means = torch.stack([x[g == i].mean(0) for i in range(int(g.max()) + 1)])
    return x - means[g]


def dsc(x, groups, tau=TAU):
    """Corrected descriptors of a collection; unchanged (only normalised) when the gate is closed."""
    return centre_by_group(x, groups) if gate_open(groups, tau) else F.normalize(x.float(), dim=1)


class DSC:
    """Fit on a collection, then correct the collection and new queries with the same group means.

        model, proc = load_siglip()
        words = word_embeddings(model, proc)
        d = DSC().fit(x_db, siglip_db, words)
        z_db, z_q = d.transform(x_db), d.transform_queries(x_q, siglip_q)
    """

    def __init__(self, min_group=MIN_GROUP, tau=TAU):
        self.min_group, self.tau = min_group, tau

    def fit(self, x, image_emb, word_emb):
        self.word_emb = word_emb
        self.groups = name_depictions(image_emb, word_emb, self.min_group)
        self.active = bool(gate_open(self.groups, self.tau))
        x = F.normalize(x.float(), dim=1)
        self.means = {int(w): x[torch.tensor(self.groups == w)].mean(0) for w in np.unique(self.groups)}
        return self

    def transform(self, x):
        x = F.normalize(x.float(), dim=1)
        if not self.active:
            return x
        return x - torch.stack([self.means[int(w)] for w in self.groups])

    def transform_queries(self, xq, image_emb_q):
        xq = F.normalize(xq.float(), dim=1)
        if not self.active:
            return xq
        words = (F.normalize(image_emb_q.float(), dim=1) @ self.word_emb.float().T).argmax(1).numpy()
        fallback = self.means.get(MERGED, torch.zeros_like(xq[0]))
        return xq - torch.stack([self.means.get(int(w), fallback) for w in words])
