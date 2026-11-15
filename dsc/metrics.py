"""Retrieval metrics of the paper, computed on the GPU when available and checked against brute force (tests/).

Collection protocol (CBIR): leave-one-out on the whole collection, cosine, relevant = same class whatever the
depiction, metrics averaged per class. Mixed collections add cross-depiction mAP and P@10, within-depiction mAP and
the depiction capture. Literature protocol: query depiction A against gallery depiction B, P@k of the official UCDIR
code (relevant in the top k over min(k, number of relevant), pooled over the queries of a task).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

TOP_KS = (10, 20, 30, 40, 50)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _codes(values):
    return np.unique(np.asarray(values), return_inverse=True)[1]


def _ap(rel, keep):
    """AP over the ranking restricted to `keep`; positions outside `keep` do not count in the rank."""
    relk = rel & keep
    rank = keep.float().cumsum(1).clamp(min=1)
    hits = relk.float().cumsum(1)
    n = relk.sum(1).float()
    ap = (relk.float() * hits / rank).sum(1) / n.clamp(min=1)
    return torch.where(n > 0, ap, torch.full_like(ap, float("nan")))


@torch.no_grad()
def per_query(x, labels, domains=None, chunk=256, device=DEVICE):
    """Per-query P@10..P@50 and AP (%), plus cross-depiction AP and P@10 when domains are given."""
    x = F.normalize(x.float().to(device), dim=1)
    lab = torch.tensor(_codes(labels), device=device)
    mixed = domains is not None
    cols = {**{f"P@{k}": [] for k in TOP_KS}, "AP": []}
    if mixed:
        dom = torch.tensor(_codes(domains), device=device)
        cols.update({"AP cross": [], "P@10 cross": []})
    for s in range(0, len(x), chunk):
        sim = x[s:s + chunk] @ x.T
        r = torch.arange(sim.shape[0], device=device)
        sim[r, r + s] = float("-inf")
        order = sim.argsort(1, descending=True)[:, :-1]          # the query itself is last and dropped
        same_c = lab[order] == lab[s:s + chunk, None]
        for k in TOP_KS:
            cols[f"P@{k}"].append(same_c[:, :k].float().mean(1))
        cols["AP"].append(_ap(same_c, torch.ones_like(same_c)))
        if mixed:
            same_d = dom[order] == dom[s:s + chunk, None]
            cross = same_c & ~same_d
            cols["AP cross"].append(_ap(cross, ~(same_c & same_d)))
            cols["P@10 cross"].append(cross[:, :10].float().mean(1))
    pq = pd.DataFrame({c: torch.cat(v).cpu().numpy() * 100 for c, v in cols.items()})
    pq.insert(0, "class", list(labels))
    return pq


def macro(pq):
    """Per-class averaging (nan-skipping), rounded to two decimals."""
    m = pq.groupby("class").mean(numeric_only=True).mean()
    return {k: round(float(v), 2) for k, v in m.items()}


def evaluate(x, labels, domains=None, **kw):
    out = macro(per_query(x, labels, domains, **kw))
    return {("mAP" if k == "AP" else k): v for k, v in out.items()}


@torch.no_grad()
def depiction_bias(x, labels, domains, chunk=256, k=10, device=DEVICE):
    """Within- and cross-depiction mAP, their gap, and the depiction capture of the top k (1 = no bias)."""
    x = F.normalize(x.float().to(device), dim=1)
    lab, dom = torch.tensor(_codes(labels), device=device), torch.tensor(_codes(domains), device=device)
    n = len(lab)
    n_cls, n_dom = torch.bincount(lab).float(), torch.bincount(dom).float()
    n_cd = torch.zeros(int(lab.max()) + 1, int(dom.max()) + 1, device=device)
    n_cd.index_put_((lab, dom), torch.ones(n, device=device), accumulate=True)
    cols = {c: [] for c in ("AP intra", "AP inter", "err", "err same", "base rate")}
    for s in range(0, n, chunk):
        sim = x[s:s + chunk] @ x.T
        r = torch.arange(sim.shape[0], device=device)
        sim[r, r + s] = float("-inf")
        order = sim.argsort(1, descending=True)[:, :-1]
        lq, dq = lab[s:s + chunk], dom[s:s + chunk]
        same_c, same_d = lab[order] == lq[:, None], dom[order] == dq[:, None]
        cols["AP inter"].append(_ap(same_c & ~same_d, ~(same_c & same_d)))
        cols["AP intra"].append(_ap(same_c & same_d, ~(same_c & ~same_d)))
        err = ~same_c[:, :k]
        cols["err"].append(err.float().sum(1))
        cols["err same"].append((err & same_d[:, :k]).float().sum(1))
        cols["base rate"].append((n_dom[dq] - n_cd[lq, dq]) / (n - n_cls[lq]))
    pq = pd.DataFrame({c: torch.cat(v).cpu().numpy() for c, v in cols.items()})
    pq["AP intra"] *= 100
    pq["AP inter"] *= 100
    pq.insert(0, "class", list(labels))
    m = pq.groupby("class")[["AP intra", "AP inter"]].mean().mean()
    w = pq["err"].sum()
    capture = (pq["err same"].sum() / w) / ((pq["err"] * pq["base rate"]).sum() / w) if w else float("nan")
    return {"mAP intra": round(float(m["AP intra"]), 2), "mAP inter": round(float(m["AP inter"]), 2),
            "gap intra-inter": round(float(m["AP intra"] - m["AP inter"]), 2), "capture": round(float(capture), 2)}


@torch.no_grad()
def pair_precision(xa, xb, la, lb, ks, chunk=1024, device=DEVICE):
    """Query depiction A -> gallery depiction B, P@k as in the official UCDIR code, pooled over the queries."""
    xa, xb = F.normalize(xa.float().to(device), dim=1), F.normalize(xb.float().to(device), dim=1)
    inv = _codes(np.concatenate([la, lb]))
    la_t, lb_t = torch.tensor(inv[:len(la)], device=device), torch.tensor(inv[len(la):], device=device)
    n_rel = torch.bincount(lb_t, minlength=int(inv.max()) + 1)[la_t]
    kmax = min(max(ks), len(xb))
    hits, den = {k: 0.0 for k in ks}, {k: 0.0 for k in ks}
    for s in range(0, len(xa), chunk):
        idx = (xa[s:s + chunk] @ xb.T).topk(kmax, dim=1).indices
        rel = lb_t[idx] == la_t[s:s + chunk, None]
        for k in ks:
            hits[k] += float(rel[:, :k].sum())
            den[k] += float(n_rel[s:s + chunk].clamp(max=k).sum())
    return {f"P@{k}": round(100 * hits[k] / den[k], 2) if den[k] else float("nan") for k in ks}
