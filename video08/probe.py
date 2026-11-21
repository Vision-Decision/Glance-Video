"""Linear probes on question-free stream features (pooled frame-group hidden states, from evals/deploy.py npz files).

python -m video08.probe --train 'DIR/stream*_none_fps2.npz' --test 'DIR/stream*_stream_fps2.npz' --out OUT.json
Targets per readout time: 'now' (class active at t), 'sofar' (class started earlier in the current clip), elapsed time in
the current clip (regression, R^2). Train streams come from Charades train (Video-Index ids removed); test streams from
Charades test. Also reports the generative yes/no AUROC on the same (stream, rule, time) cells for comparison.
"""
import argparse, glob, json
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from video08 import metrics as M

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True); ap.add_argument("--test", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--min_pos", type=int, default=20); ap.add_argument("--C", type=float, default=0.1)
a = ap.parse_args()


def load(pat):
    X, Ynow, Ysof, el, P, rules, C = {}, [], [], [], [], [], None
    for f in sorted(glob.glob(pat)):
        z = np.load(f, allow_pickle=True); C = list(z["classes"]); t = z["t"]; seg = json.loads(str(z["seg"]))
        ivs = json.loads(str(z["ivs"]))
        for k in z.files:
            if k.startswith("h"):
                X.setdefault(k, []).append(z[k].astype(np.float32)[: len(t)])
        Ynow.append(z["Y"])
        so = np.zeros_like(z["Y"]); e = np.zeros(len(t))
        for v, a0, a1 in seg:
            m = (t >= a0) & (t < a1); e[m] = t[m] - a0
            for j, c in enumerate(C):
                for s, _ in ivs[c]:
                    if a0 <= s < a1:
                        so[m & (t >= s), j] = 1
        Ysof.append(so); el.append(e)
        if z["p"].size:
            P.append((z["p"], [C.index(c) for c in z["rules"]], len(Ynow) - 1))
    return {k: np.concatenate(v) for k, v in X.items()}, Ynow, Ysof, el, P, C


Xtr, Ntr, Str, Etr, _, C = load(a.train)
Xte, Nte, Ste, Ete, Pte, _ = load(a.test)
Ntr, Str, Etr = np.concatenate(Ntr), np.concatenate(Str), np.concatenate(Etr)
NteA, SteA, EteA = np.concatenate(Nte), np.concatenate(Ste), np.concatenate(Ete)
out = dict(n_train=int(len(Ntr)), n_test=int(len(NteA)))
# generative reference on the rule cells
if Pte:
    offs = np.cumsum([0] + [len(x) for x in Nte])
    gy, gs, cells = [], [], []
    for p, cols, si in Pte:
        for j, c in enumerate(cols):
            gy.append(Nte[si][:, c]); gs.append(p[:, j]); cells.append((si, c))
    gy, gs = np.concatenate(gy), np.concatenate(gs)
    out["gen_now_auroc_rule_cells"] = M.auroc(gy, gs)
for k in sorted(Xtr, key=lambda s: int(s[1:])):
    sc = StandardScaler().fit(Xtr[k]); A_, B_ = sc.transform(Xtr[k]), sc.transform(Xte[k])
    res = {}
    for name, Ytr, Yte in (("now", Ntr, NteA), ("sofar", Str, SteA)):
        aucs, probe_scores = {}, {}
        for j, c in enumerate(C):
            if Ytr[:, j].sum() < a.min_pos or Yte[:, j].sum() < 5 or Yte[:, j].all():
                continue
            clf = LogisticRegression(C=a.C, max_iter=300, class_weight="balanced").fit(A_, Ytr[:, j])
            s = clf.decision_function(B_); aucs[c] = M.auroc(Yte[:, j], s); probe_scores[j] = s
        res[name] = dict(macro_auroc=float(np.mean(list(aucs.values()))), n_classes=len(aucs))
        if name == "now" and Pte:  # probe on exactly the generative rule cells
            py, ps = [], []
            for p, cols, si in Pte:
                for c in cols:
                    if c in probe_scores:
                        py.append(Nte[si][:, c]); ps.append(probe_scores[c][offs[si]:offs[si + 1]])
            if py:
                res[name]["probe_auroc_rule_cells"] = M.auroc(np.concatenate(py), np.concatenate(ps))
    r = Ridge(alpha=10.0).fit(A_, Etr); pe = r.predict(B_)
    res["elapsed_r2"] = float(1 - ((pe - EteA) ** 2).sum() / ((EteA - EteA.mean()) ** 2).sum())
    res["elapsed_mae_s"] = float(np.abs(pe - EteA).mean())
    out[k] = res
    print(k, json.dumps(res), flush=True)
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps(out, indent=1))
