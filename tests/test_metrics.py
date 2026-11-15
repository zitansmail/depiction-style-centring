"""Every metric against a brute-force loop on random data."""
import numpy as np
import torch
import torch.nn.functional as F

from dsc.metrics import TOP_KS, depiction_bias, pair_precision, per_query


def _data(n=240, d=16, n_cls=6, n_dom=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    lab = np.array([f"c{i}" for i in torch.randint(0, n_cls, (n,), generator=g).tolist()])
    dom = np.array([f"d{i}" for i in torch.randint(0, n_dom, (n,), generator=g).tolist()])
    return x, lab, dom


def _ap(order, rel, keep):
    hits, rank, tot = 0, 0, 0.0
    for j in order:
        if keep(j):
            rank += 1
            if rel(j):
                hits += 1; tot += hits / rank
    return 100 * tot / hits if hits else float("nan")


def _orders(x):
    xn = F.normalize(x.double(), dim=1).numpy()
    for q in range(len(xn)):
        sim = xn @ xn[q]; sim[q] = -np.inf
        yield q, [j for j in np.argsort(-sim, kind="stable") if j != q]


def test_per_query_matches_brute_force():
    x, lab, dom = _data()
    pq = per_query(x, lab, dom, chunk=64, device="cpu")
    for q, order in _orders(x):
        c, d = lab[q], dom[q]
        for k in TOP_KS:
            assert abs(pq.loc[q, f"P@{k}"] - 100 * np.mean([lab[j] == c for j in order[:k]])) < 1e-3
        assert abs(pq.loc[q, "AP"] - _ap(order, lambda j: lab[j] == c, lambda j: True)) < 1e-3
        a = _ap(order, lambda j: lab[j] == c and dom[j] != d, lambda j: not (lab[j] == c and dom[j] == d))
        assert np.isclose(pq.loc[q, "AP cross"], a, atol=1e-3, equal_nan=True)
        assert abs(pq.loc[q, "P@10 cross"] - 100 * np.mean([lab[j] == c and dom[j] != d for j in order[:10]])) < 1e-3


def test_depiction_bias_matches_brute_force():
    x, lab, dom = _data()
    out = depiction_bias(x, lab, dom, chunk=64, device="cpu")
    intra, inter, err, err_same, base = {}, {}, 0.0, 0.0, 0.0
    for q, order in _orders(x):
        c, d = lab[q], dom[q]
        intra.setdefault(c, []).append(_ap(order, lambda j: lab[j] == c and dom[j] == d,
                                           lambda j: not (lab[j] == c and dom[j] != d)))
        inter.setdefault(c, []).append(_ap(order, lambda j: lab[j] == c and dom[j] != d,
                                           lambda j: not (lab[j] == c and dom[j] == d)))
        e = [j for j in order[:10] if lab[j] != c]
        others = [j for j in range(len(lab)) if lab[j] != c]
        err += len(e); err_same += sum(dom[j] == d for j in e)
        base += len(e) * np.mean([dom[j] == d for j in others])
    m = lambda dd: np.nanmean([np.nanmean(v) for v in dd.values()])
    assert abs(out["mAP intra"] - m(intra)) < 0.01 and abs(out["mAP inter"] - m(inter)) < 0.01
    assert abs(out["capture"] - (err_same / err) / (base / err)) < 0.01


def test_pair_precision_matches_brute_force():
    x, lab, _ = _data()
    a, b, la, lb = x[:100], x[100:], lab[:100], lab[100:]
    out = pair_precision(a, b, la, lb, (1, 5, 15), chunk=32, device="cpu")
    an, bn = F.normalize(a.double(), dim=1).numpy(), F.normalize(b.double(), dim=1).numpy()
    for k in (1, 5, 15):
        hits = den = 0
        for q in range(len(an)):
            top = np.argsort(-(bn @ an[q]), kind="stable")[:k]
            hits += sum(lb[j] == la[q] for j in top); den += min(k, int((lb == la[q]).sum()))
        assert abs(out[f"P@{k}"] - round(100 * hits / den, 2)) < 0.011
