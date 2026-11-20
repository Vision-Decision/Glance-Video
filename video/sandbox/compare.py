#!/usr/bin/env python3
"""Paired comparison of two configs on their common finished items.

  python3 compare.py --a full1 --b ds_vdm_hard [--model deepseek-v4.1-flash] [--model-b ...] [--set hard60]
  python3 compare.py --a run1,run2,run3 --b vdm1,vdm2,vdm3          # several runs per config: item score = mean
Reports accuracy of each config on the common items, Δ = B − A with a paired bootstrap 95% CI (10,000 resamples
over items), a sign-flip permutation p-value, McNemar counts (fixed / broken, majority over runs), and
no-answer / time per item. Rate-limited items have no result.json and are excluded automatically.
"""
import argparse, json, os, random, statistics as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNS = os.path.join(os.environ.get("VI_DATA", os.path.join(ROOT, "data/video_index")), "runs")


def load(runs, model):
    out = {}
    for run in runs.split(","):
        d = os.path.join(RUNS, run, model.replace(":", "_"))
        if not os.path.isdir(d):
            continue
        for iid in os.listdir(d):
            f = os.path.join(d, iid, "result.json")
            if os.path.exists(f):
                out.setdefault(iid, []).append(json.load(open(f)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--model-b", default="")
    ap.add_argument("--set", default="")
    ap.add_argument("--json", default="")
    x = ap.parse_args()
    A, B = load(x.a, x.model), load(x.b, x.model_b or x.model)
    ids = sorted(set(A) & set(B))
    if x.set:
        keep = set(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{x.set}.json"))))
        ids = [i for i in ids if i in keep]
    n = len(ids)
    if not n:
        return print("no common items")
    sa = [st.mean(r["correct"] for r in A[i]) for i in ids]
    sb = [st.mean(r["correct"] for r in B[i]) for i in ids]
    d = [b - a for a, b in zip(sa, sb)]
    rng = random.Random(0)
    boots = sorted(st.mean(d[rng.randrange(n)] for _ in range(n)) for _ in range(10000))
    obs = st.mean(d)
    flips = sum(abs(st.mean(v if rng.random() < .5 else -v for v in d)) >= abs(obs) - 1e-12 for _ in range(10000))
    fixed = sum(a < .5 <= b for a, b in zip(sa, sb))
    broken = sum(b < .5 <= a for a, b in zip(sa, sb))

    def meta(R):
        rs = [r for i in ids for r in R[i]]
        return {"no_answer": sum(r["status"] != "succeeded" for r in rs),
                "timeouts": sum(r["status"] == "timeout" for r in rs),
                "min_per_item": round(st.mean(min(r["duration_s"], 3600) for r in rs) / 60, 1),
                "runs_per_item": round(len(rs) / n, 2)}
    out = {"a": x.a, "b": x.b, "n_common": n, "acc_a": round(100 * st.mean(sa), 1), "acc_b": round(100 * st.mean(sb), 1),
           "delta_pp": round(100 * obs, 1), "ci95_pp": [round(100 * boots[249], 1), round(100 * boots[9749], 1)],
           "p_signflip": round(flips / 10000, 4), "mcnemar_fixed": fixed, "mcnemar_broken": broken,
           "meta_a": meta(A), "meta_b": meta(B)}
    print(json.dumps(out, indent=1))
    if x.json:
        json.dump(out, open(x.json, "w"), indent=1)


if __name__ == "__main__":
    main()
