"""P3 summary: heads vs the same backbone's generative answers (same stream state, same tokens), paired bootstrap by video.
python -m video08.p3_report EVAL.jsonl [...]"""
import json, sys
import numpy as np
from video08 import metrics as M


def matched(y, head, gen):
    """compare at the SAME acceptance rate: thresholds so both accept the fraction of items that are positive"""
    r = y.mean(); th = np.quantile(head, 1 - r); tg = np.quantile(gen, 1 - r)
    ah, ag = head >= th, gen >= tg
    f = lambda a: dict(acc_rate=float(a.mean()), recall=float(a[y == 1].mean()), fpr=float(a[y == 0].mean()), precision=float(y[a].mean()) if a.any() else float("nan"))
    return f(ah), f(ag)


def summarize(path):
    R = [json.loads(l) for l in open(path)]
    out = {}
    P = [r for r in R if r["task"] == "presence"]
    for kind in ("now", "sofar", "all"):
        S = [r for r in P if kind == "all" or r["kind"] == kind]
        y = np.array([r["y"] for r in S]); h = np.array([r["head"] for r in S]); g = np.array([r["gen"] for r in S])
        grp = np.array([r["vid"] for r in S])
        d = M.paired_bootstrap(lambda i: M.auroc(y[i], g[i]), lambda i: M.auroc(y[i], h[i]), len(y), B=1000, groups=grp)
        mh, mg = matched(y, h, g)
        hard = np.array([r["neg"] in ("pos", "hard", "temporal") for r in S])
        o = dict(n=len(S), pos_rate=float(y.mean()), auroc_head=M.auroc(y, h), auroc_gen=M.auroc(y, g),
                 d_auroc=d[0], d_ci=[d[1], d[2]], p=d[3], ece_head=M.ece(y, h), ece_gen=M.ece(y, g),
                 fpr_r80_head=M.fpr_at_recall(y, h), fpr_r80_gen=M.fpr_at_recall(y, g),
                 acc05_head=float(((h >= .5) == y).mean()), acc05_gen=float(((g >= .5) == y).mean()),
                 matched_head=mh, matched_gen=mg,
                 auroc_hardneg_head=M.auroc(y[hard], h[hard]), auroc_hardneg_gen=M.auroc(y[hard], g[hard]))
        for nt in ("temporal", "hard", "random"):
            m = np.array([r["neg"] in ("pos", nt) for r in S])
            o[f"auroc_vs_{nt}_head"] = M.auroc(y[m], h[m]); o[f"auroc_vs_{nt}_gen"] = M.auroc(y[m], g[m])
        out[f"presence_{kind}"] = o
    T = [r for r in R if r["task"] == "temporal" and r["cls"] == "sta" and r.get("gen") is not None or (r["task"] == "temporal" and r["cls"] == "sta" and "gen_text" in r)]
    if T:
        gt = [tuple(r["segs"][0]) for r in T]
        hp = [max(r["head"], key=lambda x: x[0])[1:] for r in T]
        gp = [tuple(r["gen"]) if r.get("gen") else None for r in T]
        gh, ih = M.grounding(hp, gt); gg, ig = M.grounding(gp, gt)
        grp = np.array([r["vid"] for r in T])
        d = M.paired_bootstrap(lambda i: ig[i].mean() * 100, lambda i: ih[i].mean() * 100, len(T), B=1000, groups=grp)
        out["sta"] = dict(n=len(T), head=gh, gen=gg, d_miou=d[0], d_ci=[d[1], d[2]], p=d[3])
    A = [r for r in R if r["task"] == "temporal" and r["cls"] != "sta" and not r["segs"]]
    if A:  # absent-class temporal queries: does the head say "no segment"?
        out["temporal_absent_reject_head"] = float(np.mean([max(x[0] for x in r["head"]) < 0.5 for r in A]))
        gA = [r for r in A if "gen_text" in r]
        if gA:
            out["temporal_absent_gen_gives_span"] = float(np.mean([r.get("gen") is not None for r in gA]))
    C = [r for r in R if r["task"] == "count"]
    if C:
        y = np.array([r["y"] for r in C]); hp = np.array([int(np.argmax(r["head"])) for r in C])
        gp = np.array([r["gen"] if r.get("gen") is not None else -1 for r in C])
        grp = np.array([r["vid"] for r in C])
        obo = lambda p: (lambda i: (np.abs(p[i] - y[i]) <= 1).mean() * 100)
        d = M.paired_bootstrap(obo(gp), obo(hp), len(C), B=1000, groups=grp)
        from collections import Counter
        out["count"] = dict(n=len(C), head=M.counting(list(hp), list(y)), gen=M.counting([None if x < 0 else int(x) for x in gp], list(y)),
                            d_obo=d[0], d_ci=[d[1], d[2]], p=d[3], gen_top_answers=Counter(gp.tolist()).most_common(5),
                            head_top_answers=Counter(hp.tolist()).most_common(5))
    W = [r for r in R if r["task"] == "choice"]
    if W:
        y = np.array([r["y"] for r in W]); hp = np.array([int(np.argmax(r["head"])) for r in W]); gp = np.array([int(np.argmax(r["gen"])) for r in W])
        grp = np.array([r["vid"] for r in W])
        d = M.paired_bootstrap(lambda i: (gp[i] == y[i]).mean() * 100, lambda i: (hp[i] == y[i]).mean() * 100, len(W), B=1000, groups=grp)
        out["choice"] = dict(n=len(W), acc_head=float((hp == y).mean() * 100), acc_gen=float((gp == y).mean() * 100), d=d[0], d_ci=[d[1], d[2]], p=d[3])
    # latency per readout (one forward for all N questions) vs generative
    rd = {}
    for r in R:
        rd[(r["vid"], r["t"])] = (r["nq"], r["head_ms"], r["genlogit_ms"])
    nq = np.array([v[0] for v in rd.values()]); hm = np.array([v[1] for v in rd.values()]); gm = np.array([v[2] for v in rd.values()])
    gen_ms = [r["gen_ms"] for r in R if "gen_ms" in r]
    out["latency"] = dict(readouts=len(rd), nq_median=float(np.median(nq)), head_ms_p50=float(np.median(hm)), yesno_logit_ms_p50=float(np.median(gm)),
                          gen_decode_ms_per_question_p50=float(np.median(gen_ms)) if gen_ms else None,
                          gen_decode_ms_for_nq_p50=float(np.median(gen_ms) * np.median(nq)) if gen_ms else None)
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print("=====", p.split("/")[-1]); print(json.dumps(summarize(p), indent=1, default=float))
