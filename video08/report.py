"""Summaries for the P2 / P3 outputs.  python -m video08.report {m6,m7,m8,deploy,p3} PATH... (prints JSON)"""
import glob, json, os, sys
import numpy as np
from video08 import metrics as M


def load_jsonl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


# ------------------------------------------------------------------ M6
def m6(paths):
    out = {}
    for p in paths:
        R = load_jsonl(p); name = os.path.basename(p).replace(".jsonl", "")
        if not R:
            continue
        res = [dict(x, vid=r["id"]) for r in R for x in r["res"]]
        if "pushup" in name or (res and "gt" in res[0] and isinstance(res[0]["gt"], int)):
            pr = [M.parse_int(x["ans"]) for x in res]; gt = [x["gt"] for x in res]
            out[name] = dict(n=len(res), **M.counting(pr, gt))
        elif res and res[0]["kind"] == "gen":
            pr = [M.parse_span(x["ans"]) for x in res]; gt = [tuple(x["gt"]) for x in res]
            g, ious = M.grounding(pr, gt); out[name] = dict(n=len(res), **g)
        else:
            y = np.array([x["y"] for x in res]); s = np.array([x["p"] for x in res])
            hard = np.array([x.get("hard", 0) for x in res])
            m = (y == 1) | (hard == 1)
            out[name] = dict(n=len(res), auroc=M.auroc(y, s), auroc_hard=M.auroc(y[m], s[m]), ece=M.ece(y, s),
                             acc05=float(((s >= .5) == y).mean()), fpr_r80=M.fpr_at_recall(y, s, .8), mean_p=float(s.mean()))
        out[name]["wall_s_per_video"] = float(np.mean([r["wall_s"] for r in R]))
        out[name]["kv_tokens"] = float(np.mean([r.get("kv", 0) for r in R]))
    return out


def m6_pair(pa, pb, metric):
    """paired bootstrap over videos between two M6 runs of the same bench"""
    A = {r["id"]: r for r in load_jsonl(pa)}; B = {r["id"]: r for r in load_jsonl(pb)}
    ids = sorted(set(A) & set(B))
    ra = [x for i in ids for x in A[i]["res"]]; rb = [x for i in ids for x in B[i]["res"]]
    grp = np.array([i for i in ids for _ in A[i]["res"]])
    if metric == "auroc":
        ya = np.array([x["y"] for x in ra]); sa = np.array([x["p"] for x in ra]); sb = np.array([x["p"] for x in rb])
        f = lambda s: (lambda idx: M.auroc(ya[idx], s[idx]))
        return M.paired_bootstrap(f(sa), f(sb), len(ya), groups=grp)
    if metric == "miou":
        ia = np.array([M.tiou(M.parse_span(x["ans"]) or (0, 0), x["gt"]) for x in ra]); ib = np.array([M.tiou(M.parse_span(x["ans"]) or (0, 0), x["gt"]) for x in rb])
        return M.paired_bootstrap(lambda i: ia[i].mean() * 100, lambda i: ib[i].mean() * 100, len(ia), groups=grp)
    if metric in ("exact", "obo"):
        g = np.array([x["gt"] for x in ra]); pa_ = np.array([M.parse_int(x["ans"]) or -1 for x in ra]); pb_ = np.array([M.parse_int(x["ans"]) or -1 for x in rb])
        tol = 0 if metric == "exact" else 1
        return M.paired_bootstrap(lambda i: (np.abs(pa_[i] - g[i]) <= tol).mean() * 100, lambda i: (np.abs(pb_[i] - g[i]) <= tol).mean() * 100, len(g), groups=grp)


# ------------------------------------------------------------------ M7
def m7(paths):
    out = {}
    for p in paths:
        R = load_jsonl(p)
        if not R:
            continue
        name = os.path.basename(p).replace(".jsonl", ""); curve = {}
        deltas = sorted({x["delta"] for r in R for x in r["res"]})
        for d in deltas:
            pos, neg, tr = [], [], []
            for r in R:
                x = next((x for x in r["res"] if x["delta"] == d), None)
                if x:
                    pos.append(x["p"][0]); neg.append(x["p"][1:]); tr.append(r["trial"])
            pos = np.array(pos); neg = np.array(neg)
            y = np.r_[np.ones(len(pos)), np.zeros(neg.size)]; s = np.r_[pos, neg.ravel()]
            # per-trial 4-way "did it pick the shown word" accuracy
            acc = float((pos > neg.max(1)).mean())
            n = len(pos)
            def f(idx):
                return M.auroc(np.r_[np.ones(len(idx)), np.zeros(3 * len(idx))], np.r_[pos[idx], neg[idx].ravel()])
            lo, hi = M.bootstrap_ci(f, n, B=500)
            curve[d] = dict(n=n, auroc=M.auroc(y, s), lo=lo, hi=hi, acc4=acc, p_pos=float(pos.mean()), p_neg=float(neg.mean()))
        out[name] = curve
    return out


# ------------------------------------------------------------------ M8
def m8(paths):
    out = {}
    for p in paths:
        L = json.load(open(p)); name = os.path.basename(p).replace(".json", "")
        pr = L["probes"]; bins = {}
        for x in pr:
            b = int(x["t"] // 600)
            bins.setdefault(b, []).append(x)
        rows = {}
        for b, xs in sorted(bins.items()):
            wp = np.array([x["word_p"] for x in xs]); ay = np.concatenate([x["act_y"] for x in xs]); ap_ = np.concatenate([x["act_p"] for x in xs])
            rows[f"{b * 10}-{b * 10 + 10}min"] = dict(n=len(xs), word_auroc=M.auroc(np.r_[np.ones(len(wp)), np.zeros(wp[:, 1:].size)], np.r_[wp[:, 0], wp[:, 1:].ravel()]),
                                                     word_acc4=float((wp[:, 0] > wp[:, 1:].max(1)).mean()), act_auroc=M.auroc(ay, ap_))
        o = dict(bins=rows, wall_s=L["wall_s"], n_probes=len(pr))
        if L["steps"]:
            S = np.array(L["steps"])  # t, ms, n_groups, vit_ms/group, kv, pos, rebases, mem MiB
            per = S[:, 1] / S[:, 2]
            o.update(ms_per_group_p50=float(np.median(per)), ms_per_write_p50=float(np.median(S[:, 1])), ms_per_write_p99=float(np.percentile(S[:, 1], 99)),
                     vit_ms_per_group=float(np.median(S[:, 3])), kv_max=int(S[:, 4].max()), pos_max=int(S[:, 5].max()), rebases=int(S[-1, 6]),
                     mem_mib_first=float(S[:100, 7].mean()), mem_mib_last=float(S[-100:, 7].mean()),
                     write_ms_by_10min={int(k): float(np.median(S[(S[:, 0] // 600) == k, 1])) for k in np.unique(S[:, 0] // 600)})
        o["ask_ms_p50"] = float(np.median([x["ask_ms"] for x in pr])) if pr else None
        out[name] = o
    return out


# ------------------------------------------------------------------ deployment protocol
def deploy(paths, recall=0.8, k=2, cooldown=5.0, tol=2.0):
    """paths: npz files of ONE arm. Frame-level metrics over (stream, rule, readout); event-level FA/h at the global
    threshold that gives `recall` of events detected."""
    Z = [np.load(p, allow_pickle=True) for p in paths]
    ys, ss = [], []; ev = []
    hours = 0.0
    for z in Z:
        t = z["t"]; P = z["p"]; rules = list(z["rules"]); C = list(z["classes"]); ivs = json.loads(str(z["ivs"]))
        if P.size == 0:
            continue
        hours += (t[-1] - t[0]) / 3600
        for j, c in enumerate(rules):
            y = z["Y"][:, C.index(c)]; ys.append(y); ss.append(P[:, j])
            # merge overlapping intervals of the rule
            iv = sorted(ivs[c]); mg = []
            for s, e in iv:
                if mg and s <= mg[-1][1]:
                    mg[-1][1] = max(mg[-1][1], e)
                else:
                    mg.append([s, e])
            ev.append((t, P[:, j], mg))
    y = np.concatenate(ys); s = np.concatenate(ss)
    out = dict(n_streams=len(Z), hours=hours, n_rules=len(ev), frames=len(y), pos_rate=float(y.mean()), auroc=M.auroc(y, s),
               ece=M.ece(y, s), fpr_r80=M.fpr_at_recall(y, s, .8), fpr_r50=M.fpr_at_recall(y, s, .5))
    # event level: sweep thresholds, pick the lowest threshold giving >= recall... report FA/h and latency there
    grid = np.unique(np.quantile(s, np.linspace(0.5, 0.9999, 200)))
    best = None
    for thr in grid[::-1]:
        fa = hits = nev = 0; lat = []
        for t, sc, mg in ev:
            r = M.deploy_eval(t, sc, mg, thr, k, cooldown, tol); fa += r["fa"]; hits += r["hits"]; nev += r["n_events"]; lat += r["lat"]
        rc = hits / max(nev, 1)
        if rc >= recall:
            best = dict(thr=float(thr), event_recall=rc, fa_per_hour=fa / max(hours, 1e-9), fa_per_rule_hour=fa / max(hours, 1e-9) / max(len(ev) / len(Z), 1),
                        lat_p50=float(np.median(lat)) if lat else None, lat_p90=float(np.percentile(lat, 90)) if lat else None, n_events=nev)
            break
    out["event@R%.0f" % (recall * 100)] = best
    ask = np.concatenate([z["ask_ms"] for z in Z if z["ask_ms"].size]) if any(z["ask_ms"].size for z in Z) else np.zeros(0)
    if ask.size:
        out.update(ask_ms_p50=float(np.median(ask)), ask_ms_p99=float(np.percentile(ask, 99)))
    return out


if __name__ == "__main__":
    kind, paths = sys.argv[1], sys.argv[2:]
    fn = dict(m6=m6, m7=m7, m8=m8, deploy=deploy)[kind]
    print(json.dumps(fn(paths), indent=1, default=float))
