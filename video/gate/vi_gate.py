"""Video-Index offline gate, part 1 (CPU): build the served calibration from NON-Video-Index data and test the
set-level ops on held-out videos.

  python video/gate/vi_gate.py --pilot pilot_clips_shortgv.json.gz --scorer g_short --calib_out CALIB.json --out gate_stats.json

Data: the Charades-STA test clip pilot of the served model (scan/pilot_charades.py --unit clip --K 4 --S 2, 2 fps,
the vdmscan clip format), 700 videos, positive scans (STA query on its own video, clip labels from the action-class
instances) and clean negative scans (same query on videos without the class or its object).
Split: videos by sha1(vid) parity -> CAL (calibration served to the agent: null pool, negative scans, temperature) and
HOLD (all numbers below). A scan is in a split only if its video AND its query's source video are in it.
Checks on HOLD (alpha = q = 0.1 unless noted):
  ever   set-level false alarm on negative scans (natural length and concatenated to N = 32/64/128/256/512/900 clips),
         detection rate on positive scans
  when   FDR of BH-selected clips on positive scans (clips outside every labelled instance count as false; labels are
         coarse, so this is an upper bound), top-interval IoU with the instance (R1@0.3 / 0.5)
  count  number of reported occurrences vs labelled instances, on synthetic long scans: k positive scans of k different
         queries are not comparable, so each test scan = one positive scan with its own query embedded in n negative
         scans of the SAME query (exact: clips are scored independently); expected count = labelled runs
  auroc  clip-level AUROC (pos-scan clip labels vs neg-scan clips), with a bootstrap CI over videos
"""
import argparse, gzip, hashlib, json, math, random
import numpy as np

from scan import video_ops as VO


def split_of(v):
    return "CAL" if int(hashlib.sha1(v.encode()).hexdigest(), 16) % 2 == 0 else "HOLD"


def auroc(pos, neg):
    s = np.r_[pos, neg]; y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    # ties: average ranks
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, r); r = (sums / cnt)[inv]
    return float((r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def runs(lab):
    lab = np.asarray(lab, bool); return int(((lab[1:] & ~lab[:-1]).sum()) + lab[0]) if len(lab) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True); ap.add_argument("--scorer", default="g_short")
    ap.add_argument("--calib_out", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.1); ap.add_argument("--q", type=float, default=0.1)
    a = ap.parse_args()
    d = json.load(gzip.open(a.pilot, "rt"))
    S, fps = d["scans"], d["fps"]
    qvid = {s["qid"]: s["vid"] for s in S if s["pos"]}
    for s in S:
        s["split"] = split_of(s["vid"]) if split_of(s["vid"]) == split_of(qvid.get(s["qid"], s["vid"])) else None
        s["sc"] = np.asarray(s["scores"][a.scorer], float)
    cal = [s for s in S if s["split"] == "CAL"]
    hold = [s for s in S if s["split"] == "HOLD"]
    C = VO.Calib.from_pilot([dict(scores={a.scorer: s["sc"].tolist()}, pos=s["pos"], labels=s["labels"]) for s in cal], a.scorer,
                            name=f"charades-sta-test CAL half ({a.pilot.split('/')[-1]}, {a.scorer}, 2fps K4 S2)")
    json.dump(C.to_json(), open(a.calib_out, "w"))
    K, Sd = d["args"]["K"], d["args"]["S"]
    out = dict(scorer=a.scorer, alpha=a.alpha, q=a.q, temperature=C.T,
               n_cal=dict(scans=len(cal), neg=sum(not s["pos"] for s in cal), videos=len({s["vid"] for s in cal})),
               n_hold=dict(scans=len(hold), pos=sum(s["pos"] for s in hold), neg=sum(not s["pos"] for s in hold),
                           videos=len({s["vid"] for s in hold})))
    hp = [s for s in hold if s["pos"] and any(s["labels"])]
    hn = [s for s in hold if not s["pos"]]
    times = lambda n: [[i * Sd / fps, (i * Sd + K - 1) / fps] for i in range(n)]
    # ---- AUROC
    pos_c = np.concatenate([s["sc"][np.asarray(s["labels"], bool)] for s in hp])
    neg_c = np.concatenate([s["sc"] for s in hn])
    rng = np.random.default_rng(0)
    vids = sorted({s["vid"] for s in hold})
    bs = []
    byv = {}
    for s in hp + hn:
        byv.setdefault(s["vid"], []).append(s)
    for _ in range(200):
        pick = rng.choice(vids, len(vids))
        P = [x["sc"][np.asarray(x["labels"], bool)] for v in pick for x in byv.get(v, []) if x["pos"]]
        N = [x["sc"] for v in pick for x in byv.get(v, []) if not x["pos"]]
        if P and N:
            bs.append(auroc(np.concatenate(P), np.concatenate(N)))
    out["clip_auroc"] = dict(value=auroc(pos_c, neg_c), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    out["scan_max_auroc"] = auroc(np.array([s["sc"].max() for s in hp]), np.array([s["sc"].max() for s in hn]))
    # ---- ever: natural length + concatenated
    ev = {}
    fa_nat = [VO.ever(s["sc"], C, a.alpha)["ever"] for s in hn]
    det = [VO.ever(s["sc"], C, a.alpha)["ever"] for s in hp]
    naive_fa = [bool((s["sc"] > 0).any()) for s in hn]
    ev["natural"] = dict(n_neg=len(hn), false_alarm=float(np.mean(fa_nat)), naive_p05_false_alarm=float(np.mean(naive_fa)),
                         n_pos=len(hp), detection=float(np.mean(det)), naive_p05_detection=float(np.mean([bool((s["sc"] > 0).any()) for s in hp])),
                         median_clips=int(np.median([len(s["sc"]) for s in hn])))
    for N in (32, 64, 128, 256, 512, 900):
        fa, fan, dets = [], [], []
        order = rng.permutation(len(hn))
        buf = []
        for i in order:
            buf.append(hn[i]["sc"])
            if sum(map(len, buf)) >= N:
                x = np.concatenate(buf)[:N]; buf = []
                fa.append(VO.ever(x, C, a.alpha)["ever"]); fan.append(bool((x > 0).any()))
        for s in hp[:400]:   # positive embedded in negatives of other queries' scans (score arrays are independent)
            fill = [hn[j]["sc"] for j in rng.permutation(len(hn))[:60]]
            pre = np.concatenate(fill)[: max(0, N - len(s["sc"]))]
            k = rng.integers(0, len(pre) + 1)
            x = np.r_[pre[:k], s["sc"], pre[k:]][:max(N, len(s["sc"]))]
            dets.append(VO.ever(x, C, a.alpha)["ever"])
        ev[str(N)] = dict(n_neg=len(fa), false_alarm=float(np.mean(fa)), naive_p05_false_alarm=float(np.mean(fan)),
                          detection=float(np.mean(dets)), tau=C.tau_max(N, a.alpha))
    out["ever"] = ev
    # ---- when / count, two selection rules, on CAL (to choose) and HOLD (to report)
    cp = [s for s in cal if s["pos"] and any(s["labels"])]
    cn = [s for s in cal if not s["pos"]]
    def when_stats(P, method):
        fdp, fdp_tol, iou3, iou5, nsel = [], [], [], [], []
        for s in P:
            lab = np.asarray(s["labels"], bool)
            w = VO.when(s["sc"], times(len(lab)), C, a.q, a.alpha, method=method)
            if not w["intervals"]:
                continue
            sel = np.zeros(len(lab), bool)
            for iv in w["intervals"]:
                i0 = int(round(iv["t0"] * fps / Sd)); i1 = int(round((iv["t1"] * fps - K + 1) / Sd))
                sel[i0:i1 + 1] = True
            near = np.convolve(lab.astype(float), np.ones(5), "same") > 0       # within 2 clips (~2 s) of a label
            fdp.append(float((sel & ~lab).sum() / max(1, sel.sum())))
            fdp_tol.append(float((sel & ~near).sum() / max(1, sel.sum()))); nsel.append(int(sel.sum()))
            top = w["intervals"][0]
            idx = np.where(lab)[0]
            gt0, gt1 = (idx[0] * Sd) / fps, (idx[-1] * Sd + K - 1) / fps
            inter = max(0.0, min(top["t1"], gt1) - max(top["t0"], gt0)); uni = max(top["t1"], gt1) - min(top["t0"], gt0)
            iou = inter / uni if uni > 0 else 0.0
            iou3.append(iou >= 0.3); iou5.append(iou >= 0.5)
        return dict(n_pos=len(P), n_with_interval=len(fdp), mean_fdp=float(np.mean(fdp)), median_fdp=float(np.median(fdp)),
                    mean_fdp_tolerant_2s=float(np.mean(fdp_tol)),
                    R1_iou03=float(np.sum(iou3) / len(P)), R1_iou05=float(np.sum(iou5) / len(P)),
                    R1_iou05_given_alarm=float(np.mean(iou5)), mean_selected_clips=float(np.mean(nsel)))
    def count_stats(P, Nn, method):
        negs_by_q = {}
        for s in Nn:
            negs_by_q.setdefault(s["qid"], []).append(s["sc"])
        rr = np.random.default_rng(1)
        rows = []
        for s in P:
            ng = negs_by_q.get(s["qid"], [])
            if len(ng) < 2:
                continue
            parts = [s["sc"]] + ng
            labs = [np.asarray(s["labels"], bool)] + [np.zeros(len(x), bool) for x in ng]
            o = rr.permutation(len(parts))
            x = np.concatenate([parts[i] for i in o]); y = np.concatenate([labs[i] for i in o])
            c = VO.count(x, times(len(x)), C, a.q, a.alpha, method=method)["count"]
            rows.append((runs(y), c, len(x)))
        gt = np.array([r[0] for r in rows]); pr = np.array([r[1] for r in rows])
        return dict(n=len(rows), mean_len_clips=float(np.mean([r[2] for r in rows])), gt_mean=float(gt.mean()),
                    exact=float((gt == pr).mean()), within1=float((abs(gt - pr) <= 1).mean()),
                    mae=float(abs(gt - pr).mean()), over=float((pr > gt).mean()), under=float((pr < gt).mean()),
                    pred_hist={str(k): int((pr == k).sum()) for k in range(0, 6)})
    out["when"] = {f"{sp}_{m}": when_stats(P, m) for sp, P in (("CAL", cp), ("HOLD", hp)) for m in ("bh", "fwer")}
    out["count_synthetic"] = {f"{sp}_{m}": count_stats(P, Nn, m) for sp, P, Nn in (("CAL", cp, cn), ("HOLD", hp, hn)) for m in ("bh", "fwer")}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
