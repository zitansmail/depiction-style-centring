"""Reproduce the tables of the paper from scratch (frozen models, nothing trained).

    python scripts/reproduce.py main          # Tables: depiction bias, main results, single-depiction collections
    python scripts/reproduce.py literature    # comparison with published methods (query depiction A -> gallery B)
    python scripts/reproduce.py ablations     # k-means groups, minimum group size, gate threshold, ImageNet-R renditions
    python scripts/reproduce.py cost          # timings
    python scripts/reproduce.py typical       # the two typical Office-Home queries (figure)
    python scripts/reproduce.py all
Options: --cap N keeps at most N images per (class, domain) for a quick test; --out results; --cache cache.
Descriptors are cached in --cache; every table is written as CSV in --out.
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dsc import baselines as B                                       # noqa: E402
from dsc.data import MIXED, SINGLE, NAMES, load                       # noqa: E402
from dsc.features import MODELS, DEVICE, descriptors, cached, style_statistics  # noqa: E402
from dsc.method import (DEPICTIONS, TAU, load_siglip, word_embeddings, name_depictions, gate_open,  # noqa: E402
                        centre_by_group, dsc, merge_small)
from dsc.metrics import evaluate, depiction_bias, pair_precision, per_query  # noqa: E402

SEEDS = (0, 1, 2)
DN_PAIRS = [("clipart", "sketch"), ("infograph", "real"), ("infograph", "sketch"), ("painting", "clipart"),
            ("painting", "quickdraw"), ("quickdraw", "real")]


class Ctx:
    def __init__(self, args):
        self.args, self.tag = args, (f"_cap{args.cap}" if args.cap else "")
        self._siglip, self._words, self._data = None, None, {}
        os.makedirs(args.out, exist_ok=True)

    def siglip(self):
        if self._siglip is None:
            self._siglip = load_siglip(device=DEVICE)
        return self._siglip

    def words(self):
        if self._words is None:
            self._words = word_embeddings(*self.siglip(), device=DEVICE)
        return self._words

    def clair(self):
        if getattr(self, "_clair", None) is None:
            self._clair = B.clair_maps(*self.siglip(), device=DEVICE)
        return self._clair

    def data(self, name):
        if name not in self._data:
            src, labels, domains = load(name, cap=self.args.cap)
            X = descriptors(name, src, self.args.cache, self.tag)
            self._data[name] = (src, labels, domains, X)
        return self._data[name]

    def csv(self, df, name):
        path = os.path.join(self.args.out, f"{name}{self.tag}.csv")
        df.to_csv(path, index=False)
        print("wrote", path)


def run_main(ctx):
    rows, flaw = [], []
    for name in MIXED + SINGLE:
        _, labels, domains, X = ctx.data(name)
        mixed = name in MIXED
        dom = domains if mixed else None
        g = name_depictions(X["SigLIP 2"], ctx.words())
        for model in MODELS:
            x = X[model]
            add = lambda method, xv, seed=None: rows.append(
                {"collection": NAMES[name], "model": model, "method": method, "seed": seed,
                 **evaluate(xv, labels, dom)})
            add("raw", x)
            if mixed:
                add("global centring", B.global_centring(x))
                for s in SEEDS:
                    add("random groups", centre_by_group(x, B.random_groups(g, s)), s)
                if model == "SigLIP 2":
                    add("prompt projection", B.prompt_projection(x, ctx.words()))
                    add("CLAIR-style map", B.clair_apply(x, g, ctx.clair()))
                add("DSC", dsc(x, g))
                for k, v in B.oracle(x, domains).items():
                    add(k, v)
                for method, xv in (("raw", x), ("DSC", dsc(x, g))):
                    flaw.append({"collection": NAMES[name], "model": model, "method": method,
                                 **depiction_bias(xv, labels, domains)})
            else:
                add("without gate", centre_by_group(x, g))
                add("DSC", dsc(x, g))
        print(f"{NAMES[name]}: largest group {np.unique(g, return_counts=True)[1].max() / len(g):.3f}, "
              f"gate {'open' if gate_open(g) else 'closed'}")
    df = pd.DataFrame(rows)
    ctx.csv(df, "main_all_rows")
    agg = df.drop(columns="seed").groupby(["collection", "model", "method"], sort=False).mean().round(2).reset_index()
    ctx.csv(agg[agg.collection.isin([NAMES[n] for n in MIXED])], "T2_main")
    ctx.csv(agg[agg.collection.isin([NAMES[n] for n in SINGLE])].dropna(axis=1), "T3_no_cost")
    ctx.csv(pd.DataFrame(flaw), "T1_flaw")


def run_literature(ctx):
    rows = []
    for name, pairs, ks in (("officehome", None, (1, 5, 15)), ("domainnet7", DN_PAIRS, (50, 100, 200))):
        _, labels, domains, X = ctx.data(name)
        pairs = pairs or list(itertools.combinations(sorted(set(domains)), 2))
        for a, b in pairs:
            m = np.isin(domains, [a, b]); mt = torch.from_numpy(m)
            lab, dom = labels[m], domains[m]
            g = name_depictions(X["SigLIP 2"][mt], ctx.words())
            for model in MODELS:
                for method, xv in (("raw", F.normalize(X[model][mt].float(), dim=1)), ("DSC", dsc(X[model][mt], g))):
                    for q, gal in ((a, b), (b, a)):
                        iq, ig = torch.from_numpy(dom == q), torch.from_numpy(dom == gal)
                        rows.append({"collection": NAMES[name], "task": f"{q} -> {gal}", "model": model,
                                     "method": method, **pair_precision(xv[iq], xv[ig], lab[dom == q], lab[dom == gal], ks)})
    df = pd.DataFrame(rows)
    ctx.csv(df, "T4_per_task")
    ctx.csv(df.drop(columns="task").groupby(["collection", "model", "method"]).mean().round(2).reset_index(),
            "T4_literature")


def run_ablations(ctx):
    from sklearn.cluster import KMeans
    rows, rend = [], []
    for name in MIXED + SINGLE:
        src, labels, domains, X = ctx.data(name)
        dom = domains if name in MIXED else None
        for model in ("DINOv2", "Fusion"):
            for mg in (10, 50):
                g = name_depictions(X["SigLIP 2"], ctx.words(), min_group=mg)
                rows.append({"collection": NAMES[name], "model": model, "ablation": f"min group {mg}",
                             "mAP": evaluate(dsc(X[model], g), labels, dom)["mAP"]})
            for tau in (0.5, 0.6, 0.75):
                g = name_depictions(X["SigLIP 2"], ctx.words())
                rows.append({"collection": NAMES[name], "model": model, "ablation": f"gate {tau}",
                             "mAP": evaluate(dsc(X[model], g, tau), labels, dom)["mAP"]})
            if name != "imagenetr" and name != "caltech256" or ctx.args.kmeans_all:
                S = cached(f"{ctx.args.cache}/{name}{ctx.tag}_style.pt", lambda: style_statistics(src)).numpy()
                for s in SEEDS:
                    g = merge_small(KMeans(n_clusters=8, n_init=10, random_state=s).fit_predict(S))
                    rows.append({"collection": NAMES[name], "model": model, "ablation": "k-means style groups",
                                 "seed": s, "mAP": evaluate(centre_by_group(X[model], g), labels, dom)["mAP"]})
    _, labels, domains, X = ctx.data("imagenetr")                     # per rendition (figure of the paper)
    g = name_depictions(X["SigLIP 2"], ctx.words())
    for model in ("DINOv2", "Fusion"):
        variants = {"raw": X[model], "DSC": dsc(X[model], g), **B.oracle(X[model], domains)}
        ap = {k: per_query(v, labels).AP.values for k, v in variants.items()}
        words = np.array(["merged" if i < 0 else DEPICTIONS[i] for i in g])
        for d in sorted(set(domains)):
            i = domains == d
            w = pd.Series(words[i]).value_counts()
            rend.append({"model": model, "rendition": d, "images": int(i.sum()), "main word": w.index[0],
                         "% in main word": round(100 * w.iloc[0] / i.sum(), 1),
                         **{f"gain {k}": round(float(np.nanmean(a[i]) - np.nanmean(ap['raw'][i])), 2)
                            for k, a in ap.items() if k != "raw"}})
    ctx.csv(pd.DataFrame(rows), "T5_ablations")
    ctx.csv(pd.DataFrame(rend), "T5_renditions")


def run_cost(ctx):
    _, labels, _, X = ctx.data("officehome")
    m, p = ctx.siglip()
    pix = torch.randn(64, 3, 224, 224, device=DEVICE)
    sync = (lambda: torch.cuda.synchronize()) if DEVICE == "cuda" else (lambda: None)
    with torch.no_grad():
        m.get_image_features(pixel_values=pix); sync(); t0 = time.time()
        for _ in range(3):
            m.get_image_features(pixel_values=pix)
        sync(); t_sig = (time.time() - t0) / (3 * 64)
    t0 = time.time(); g = name_depictions(X["SigLIP 2"], word_embeddings(m, p, device=DEVICE)); t_grp = time.time() - t0
    t0 = time.time(); xd = dsc(X["DINOv2"], g); t_cen = time.time() - t0
    xq = F.normalize(xd, dim=1).to(DEVICE)
    sync(); t0 = time.time(); (xq[:1000] @ xq.T).topk(10, dim=1); sync(); t_search = (time.time() - t0) / 1000
    ctx.csv(pd.DataFrame([("SigLIP 2 image embedding", 1000 * t_sig, "ms / image"),
                          ("depiction groups, whole collection", t_grp, f"s / {len(labels)} images"),
                          ("per-group centring, whole collection", t_cen, f"s / {len(labels)} images"),
                          ("top-10 search per query", 1e6 * t_search, "µs / query")],
                         columns=["step", "time", "unit"]).round(3), "T6_cost")


def run_typical(ctx):
    """Art and clip-art queries of Office-Home whose AP gain (SigLIP 2) is closest to the median gain."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    src, labels, domains, X = ctx.data("officehome")
    x = F.normalize(X["SigLIP 2"].float(), dim=1)
    xd = F.normalize(dsc(X["SigLIP 2"], name_depictions(X["SigLIP 2"], ctx.words())), dim=1)
    ap_raw, ap_dsc = per_query(x, labels).AP.values, per_query(xd, labels).AP.values
    pool = np.where(np.isin(domains, ["Art", "Clipart"]))[0]
    gain = ap_dsc[pool] - ap_raw[pool]
    med = np.median(gain)
    qs = pool[np.argsort(np.abs(gain - med), kind="stable")[:2]]

    def ranks_of(xv, q):
        sim = xv @ xv[q]; sim[q] = float("-inf")
        order = sim.argsort(descending=True)[:-1].numpy()
        rank = np.empty(len(xv), dtype=int); rank[order] = np.arange(1, len(order) + 1)
        return order, rank

    rows, fig = [], plt.figure(figsize=(11, 4.6))
    gs = gridspec.GridSpec(2, 7, width_ratios=[1, 1, 1, 1, 1, 0.15, 2.6], wspace=0.25, hspace=0.45)
    for r, q in enumerate(qs):
        (o_raw, rk_raw), (o_dsc, rk_dsc) = ranks_of(x, q), ranks_of(xd, q)
        cross = (labels == labels[q]) & (domains != domains[q])
        rows.append({"query_index": int(q), "depiction": domains[q], "class": labels[q],
                     "ap_raw": round(float(ap_raw[q]), 1), "ap_dsc": round(float(ap_dsc[q]), 1),
                     "n_cross_relevant": int(cross.sum()), "pool_size": len(pool), "median_gain": round(float(med), 2)})
        ax = fig.add_subplot(gs[r, 0]); ax.imshow(src.get(int(q))); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"query: {domains[q]}\n{labels[q]}", fontsize=7)
        for c, t in enumerate([t for t in o_dsc if cross[t]][:4]):
            ax = fig.add_subplot(gs[r, c + 1]); ax.imshow(src.get(int(t))); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{domains[t]}\nrank {rk_raw[t]} → {rk_dsc[t]}", fontsize=7)
        ax = fig.add_subplot(gs[r, 6])
        for o, lab, col in ((o_raw, "Raw", "#9a9893"), (o_dsc, "DSC", "#2a78d6")):
            ax.plot(np.arange(1, 301), np.cumsum(cross[o[:300]]), color=col, lw=2, label=lab)
        ax.set_xlabel("rank r", fontsize=7); ax.set_ylabel("relevant, other depiction, in top r", fontsize=7)
        ax.legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(ctx.args.out, f"typical_queries{ctx.tag}.png"), dpi=300, bbox_inches="tight")
    ctx.csv(pd.DataFrame(rows), "typical_queries")


STAGES = {"main": run_main, "literature": run_literature, "ablations": run_ablations, "cost": run_cost,
          "typical": run_typical}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=list(STAGES) + ["all"])
    ap.add_argument("--cap", type=int, default=None, help="at most N images per (class, domain), for a quick test")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--kmeans-all", action="store_true", help="k-means ablation also on the held-out collections")
    args = ap.parse_args()
    ctx = Ctx(args)
    for stage in (STAGES if args.stage == "all" else [args.stage]):
        STAGES[stage](ctx)
