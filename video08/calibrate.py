"""Temperature scaling for presence scores, fitted on probe_val (Charades train hold-out, never evaluated on).
python -m video08.calibrate PROBEVAL.jsonl -> prints {"head": T, "gen": T, "sft_first": T}"""
import json, sys
import numpy as np


def fit(y, p):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    best = None
    for T in np.exp(np.linspace(np.log(0.2), np.log(10), 400)):
        q = 1 / (1 + np.exp(-z / T)); nll = -np.mean(y * np.log(q + 1e-9) + (1 - y) * np.log(1 - q + 1e-9))
        if best is None or nll < best[0]:
            best = (nll, T)
    return float(best[1])


def apply(p, T):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)); return 1 / (1 + np.exp(-z / T))


if __name__ == "__main__":
    R = [json.loads(l) for l in open(sys.argv[1]) if '"presence"' in l]
    y = np.array([r["y"] for r in R]); out = {}
    for k in ("head", "gen", "sft_first"):
        if k in R[0]:
            out[k] = fit(y, np.array([r[k] for r in R]))
    print(json.dumps(out))
