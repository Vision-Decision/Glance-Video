"""Side-by-side table of eval_video.py results: python compare_results.py DIR  (files <model>_<eval>.json)"""
import glob, json, os, sys
from collections import defaultdict

d = sys.argv[1]
R = defaultdict(dict)
for f in sorted(glob.glob(f"{d}/*.json")):
    b = os.path.basename(f)[:-5]
    model, ev = b.split("_", 1)
    R[ev][model] = json.load(open(f))
models = sorted({m for v in R.values() for m in v})
print("| eval | metric | " + " | ".join(models) + " |")
print("|---|---|" + "---|" * len(models))


def row(ev, metric, get):
    vals = []
    for m in models:
        try:
            v = get(R[ev][m])
            vals.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        except Exception:
            vals.append("")
    print(f"| {ev} | {metric} | " + " | ".join(vals) + " |")


if "typed" in R:
    tasks = sorted({t for m in R["typed"].values() for t in m.get("raw", {})})
    for t in tasks:
        row("typed", f"{t} acc", lambda x, t=t: x["raw"][t]["acc"])
        if t.startswith("event_") and "auroc" in next(iter(R["typed"].values()))["raw"].get(t, {}):
            row("typed", f"{t} auroc", lambda x, t=t: x["raw"][t]["auroc"])
        if t in ("aot", "camera", "count", "event_choice", "sta"):
            row("typed", f"{t} ece(cal)", lambda x, t=t: x.get("cal", x["raw"])[t]["ece"])
for k in ("abs", "rel"):
    for m in ("R1@0.5", "R1@0.7", "mIoU"):
        row("sta", f"{k} {m}", lambda x, k=k, m=m: x[k][m])
for k in ("pred_argmax", "pred_ev"):
    for m in ("MAE", "OBO"):
        row("repcount", f"{k} {m}", lambda x, k=k, m=m: x[k][m])
for ev in ("vi_v1_1", "vi_v1_0"):
    for k in ("clips8", "global", "both"):
        row(ev, f"{k} acc", lambda x, k=k: x[k]["acc"])
        row(ev, f"{k} argmin", lambda x, k=k: x[k]["argmin"])
