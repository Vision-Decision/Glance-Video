"""Metrics + paired bootstrap. All functions take plain lists / numpy arrays."""
import re
import numpy as np


def auroc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, r); r = sums[inv] / cnt[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(y, p, bins=15):
    y = np.asarray(y, float); p = np.asarray(p, float)
    e = 0.0; edges = np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & ((p < b) if b < 1 else (p <= b))
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def fpr_at_recall(y, s, recall=0.8):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    if not y.any() or y.all():
        return float("nan")
    thr = np.quantile(s[y], 1 - recall)
    return float((s[~y] >= thr).mean())


def tiou(a, b):
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0])); u = max(a[1], b[1]) - min(a[0], b[0])
    return i / u if u > 0 else 0.0


NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse_span(text):
    xs = [float(x) for x in NUM.findall(text)]
    if len(xs) < 2:
        return None
    a, b = xs[0], xs[1]
    return (min(a, b), max(a, b))


WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
                                     "fifteen sixteen seventeen eighteen nineteen twenty".split())}


def parse_int(text):
    m = NUM.search(text)
    if m:
        return int(round(float(m.group())))
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in WORDS:
            return WORDS[w]
    return None


def grounding(preds, gts):
    ious = np.array([tiou(p, g) if p is not None else 0.0 for p, g in zip(preds, gts)])
    return dict(mIoU=float(ious.mean() * 100), R03=float((ious >= .3).mean() * 100), R05=float((ious >= .5).mean() * 100),
                R07=float((ious >= .7).mean() * 100), parsed=float(np.mean([p is not None for p in preds]) * 100)), ious


def counting(preds, gts):
    p = np.array([x if x is not None else -1 for x in preds], float); g = np.array(gts, float)
    return dict(exact=float((p == g).mean() * 100), obo=float((np.abs(p - g) <= 1).mean() * 100),
                mae=float(np.mean(np.abs(np.where(p < 0, 0, p) - g) / np.maximum(g, 1))),
                parsed=float((p >= 0).mean() * 100))


def bootstrap_ci(fn, n_items, B=2000, seed=0, groups=None):
    """percentile CI of fn(idx) over item (or group) resamples"""
    rng = np.random.default_rng(seed); vals = []
    if groups is not None:
        ug = np.unique(groups); gi = {g: np.nonzero(groups == g)[0] for g in ug}
    for _ in range(B):
        if groups is None:
            idx = rng.integers(0, n_items, n_items)
        else:
            idx = np.concatenate([gi[g] for g in rng.choice(ug, len(ug))])
        vals.append(fn(idx))
    vals = np.array(vals, float)
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def paired_bootstrap(fn_a, fn_b, n_items, B=2000, seed=0, groups=None):
    """delta = fn_b - fn_a on the same resample; returns (delta, lo, hi, p_two_sided)"""
    rng = np.random.default_rng(seed); d = []
    if groups is not None:
        ug = np.unique(groups); gi = {g: np.nonzero(groups == g)[0] for g in ug}
    full = np.arange(n_items)
    d0 = fn_b(full) - fn_a(full)
    for _ in range(B):
        idx = rng.integers(0, n_items, n_items) if groups is None else np.concatenate([gi[g] for g in rng.choice(ug, len(ug))])
        d.append(fn_b(idx) - fn_a(idx))
    d = np.array(d, float); d = d[~np.isnan(d)]
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(d0), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(min(1.0, p))


# ---------------------------------------------------------------- deployment protocol (stream x N rules)
def triggers(times, scores, thr, k=2, cooldown=5.0):
    """debounce: k consecutive readouts >= thr trigger once; then silent for `cooldown` seconds"""
    out, run, last = [], 0, -1e9
    for t, s in zip(times, scores):
        run = run + 1 if s >= thr else 0
        if run >= k and t - last >= cooldown:
            out.append(t); last = t; run = 0
    return out


def deploy_eval(times, scores, intervals, thr, k=2, cooldown=5.0, tol=2.0, hours=None):
    """one rule on one stream. intervals: [(s,e)] where the rule is true. A trigger inside [s - tol, e + tol] is a hit for
    that interval (first hit gives the latency t - s); other triggers are false alarms."""
    tr = triggers(times, scores, thr, k, cooldown)
    fa, lat, hit = 0, [], [False] * len(intervals)
    for t in tr:
        j = next((i for i, (s, e) in enumerate(intervals) if s - tol <= t <= e + tol), None)
        if j is None:
            fa += 1
        elif not hit[j]:
            hit[j] = True; lat.append(max(0.0, t - intervals[j][0]))
    return dict(fa=fa, hits=sum(hit), n_events=len(intervals), lat=lat, n_trig=len(tr))
