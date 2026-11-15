"""Controls, text-based baselines and oracles compared with DSC in the paper. None of them is trained."""
import numpy as np
import torch
import torch.nn.functional as F

from .method import DEPICTIONS, centre_by_group, as_tensor


def global_centring(x):
    x = F.normalize(x.float(), dim=1)
    return x - x.mean(0)


def random_groups(groups, seed):
    """Same group sizes as DSC, random members: tests whether naming the depictions matters."""
    return np.random.RandomState(seed).permutation(groups)


def remove(x, basis):
    return x - (x @ basis) @ basis.T


def prompt_basis(word_emb):
    """Orthonormal basis of the span of the centred depiction-word embeddings (Chuang et al. style)."""
    u, s, _ = torch.linalg.svd((word_emb - word_emb.mean(0)).T, full_matrices=False)
    return u[:, s > 1e-6 * s[0]]


def prompt_projection(x, word_emb):
    return remove(F.normalize(x.float(), dim=1), prompt_basis(word_emb))


def oracle(x, domains):
    """Upper bounds with the true depiction labels: per-depiction centring and removal of the depiction subspace."""
    x = F.normalize(x.float(), dim=1)
    dom = torch.tensor(np.unique(domains, return_inverse=True)[1])
    dm = torch.stack([x[dom == d].mean(0) for d in range(int(dom.max()) + 1)])
    u, s, _ = torch.linalg.svd((dm - dm.mean(0)).T, full_matrices=False)
    return {"oracle centring": x - dm[dom], "oracle subspace": remove(x - x.mean(0), u[:, s > 1e-6 * s[0]])}


BAD_WORDS = {"shit", "fuck", "fucking", "bitch", "ass", "dick", "pussy", "cock", "whore", "slut", "porn", "nigga",
             "nigger", "cunt", "bastard", "damn", "crap", "sex", "penis", "vagina", "boob", "boobs", "tits"}


def generic_vocabulary(n):
    """The n most frequent English words that are concrete common nouns, built without looking at any dataset."""
    import nltk
    from wordfreq import top_n_list
    nltk.download("wordnet", quiet=True)
    from nltk.corpus import wordnet as wn
    phys = wn.synset("physical_entity.n.01")
    count = lambda w, pos: sum(l.count() for s in wn.synsets(w, pos) for l in s.lemmas() if l.name().lower() == w)

    def concrete_noun(w):
        if not (w.isalpha() and len(w) >= 3) or w in BAD_WORDS:
            return False
        s = wn.synsets(w)
        if not s or s[0].pos() != "n" or s[0].instance_hypernyms():
            return False
        if not any(l.name().lower() == w for l in s[0].lemmas()):
            return False
        if not any(phys in path for path in s[0].hypernym_paths()):
            return False
        return count(w, wn.NOUN) >= count(w, wn.VERB) + count(w, wn.ADJ) + count(w, wn.ADV)

    out = []
    for w in top_n_list("en", 300000):
        if concrete_noun(w):
            out.append(w)
            if len(out) == n:
                break
    return out


@torch.no_grad()
def text_embeddings(model, processor, prompts, device="cpu", bs=256):
    out = []
    for s in range(0, len(prompts), bs):
        t = processor(text=prompts[s:s + bs], padding="max_length", max_length=64, truncation=True,
                      return_tensors="pt").to(device)
        out.append(F.normalize(as_tensor(model.get_text_features(**t)).float(), dim=-1).cpu())
    return torch.cat(out)


def clair_maps(model, processor, n_nouns=3000, device="cpu"):
    """Per-word orthogonal Procrustes maps from 'a {w} of a {noun}' to 'a photo of a {noun}' (CLAIR-style)."""
    nouns = generic_vocabulary(n_nouns)
    photo = text_embeddings(model, processor, [f"a photo of a {n}." for n in nouns], device)
    maps = {}
    for i, w in enumerate(DEPICTIONS):
        if w != "photo":
            a = text_embeddings(model, processor, [f"a {w} of a {n}." for n in nouns], device)
            u, _, vt = torch.linalg.svd(photo.T @ a)
            maps[i] = u @ vt
    return maps


def clair_apply(x, groups, maps):
    x = F.normalize(x.float(), dim=1).clone()
    for i, om in maps.items():
        m = torch.tensor(np.asarray(groups) == i)
        if m.any():
            x[m] = x[m] @ om.T
    return x


__all__ = ["global_centring", "random_groups", "prompt_projection", "oracle", "clair_maps", "clair_apply",
           "centre_by_group"]
