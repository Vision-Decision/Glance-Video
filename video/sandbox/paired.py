#!/usr/bin/env python3
"""Paired comparison of two sandbox runs on the same items (e.g. no tool vs +vdmscan), for the VI tool experiment.

  python3 video/sandbox/paired.py --a ds_base_hard --b ds_scan_hard [--ref full1] [--out paired.json] [--md paired.md]

Per item (both runs finished): correct (timeout / no answer = wrong), wall time (timeouts count as the limit), planner
tokens from the gateway log and DeepSeek list-price cost-equivalent, tool calls, frame-extraction steps (Bash commands
that run ffmpeg / OpenCV / PIL to write frames, plus Read calls on image files), vdmscan calls (from the trace), timeouts.
Statistics: paired difference B - A with a 10,000-resample item bootstrap CI, exact McNemar on discordant items,
per capability group and per "counting" slice (fine_category *counting*).
"""
import argparse, collections, json, math, os, random, re, statistics as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESEARCH = os.environ.get("VI_RESEARCH", os.path.join(ROOT, "data/research/video_index_v1_1"))
RUNS = os.path.join(os.environ.get("VI_DATA", os.path.join(ROOT, "data/video_index")), "runs")
ITEMS = os.path.join(RESEARCH, "data/test.jsonl")
DPRICE = (0.30e-6, 0.006e-6, 1.20e-6)   # deepseek-flash peak list price $/token (in, cached in, out); off-peak is half
GLOG = os.environ.get("VI_GATEWAY_LOG", "usage.jsonl")


def peak(ts):
    """DeepSeek peak hours: 01-04 and 06-10 UTC, Monday-Friday (public holidays ignored)."""
    import datetime as dt
    t = dt.datetime.fromtimestamp(ts, dt.timezone.utc)
    return t.weekday() < 5 and (1 <= t.hour < 4 or 6 <= t.hour < 10)
FRAME_RE = re.compile(r"ffmpeg[^|;&]*\.(jpg|jpeg|png)|cv2\.imwrite|\.save\([^)]*\.(jpg|jpeg|png)|select=|-vframes|tile=", re.I)
IMG_RE = re.compile(r"\.(jpg|jpeg|png|webp)$", re.I)


def load(run, model):
    d = os.path.join(RUNS, run, model)
    out = {}
    for i in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        p = os.path.join(d, i, "result.json")
        if os.path.exists(p):
            r = json.load(open(p))
            r.update(trace_stats(os.path.join(d, i)))
            out[i] = r
    return out


def trace_stats(cdir):
    n_frame, n_scan, n_read_img, n_bash = 0, 0, 0, 0
    ops = collections.Counter()
    try:
        for line in open(os.path.join(cdir, "stdout.jsonl"), errors="ignore"):
            if '"tool_use"' not in line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            for c in (ev.get("message") or {}).get("content", []) if ev.get("type") == "assistant" else []:
                if c.get("type") != "tool_use":
                    continue
                inp = c.get("input") or {}
                if c.get("name") == "Bash":
                    n_bash += 1
                    cmd = str(inp.get("command", ""))
                    k = len(re.findall(r"\bvdmscan\s+(find|ever|when|count|curve|reps)\b", cmd))
                    n_scan += k
                    for m in re.findall(r"\bvdmscan\s+(find|ever|when|count|curve|reps)\b", cmd):
                        ops[m] += 1
                    if FRAME_RE.search(cmd):
                        n_frame += 1
                elif c.get("name") == "Read" and IMG_RE.search(str(inp.get("file_path", ""))):
                    n_read_img += 1
    except FileNotFoundError:
        pass
    calls = []
    p = os.path.join(cdir, "work", "scratch", "vdmscan_calls.jsonl")
    if os.path.exists(p):
        calls = [json.loads(l) for l in open(p) if l.strip()]
    return dict(n_frame_steps=n_frame + n_read_img, n_frame_cmds=n_frame, n_read_img=n_read_img, n_scan_cmds=n_scan,
                scan_ops=dict(ops), scan_calls=calls)


def tokens(runs):
    tok = collections.defaultdict(collections.Counter)
    glog = os.path.join(RUNS, "_gateway", GLOG)
    for l in open(glog):
        try:
            r = json.loads(l)
        except Exception:
            continue
        run, _, iid = str(r.get("call", "")).partition(".")
        if run not in runs:
            continue
        u = r.get("usage") or {}
        t = tok[(run, iid)]
        t["requests"] += 1
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens"):
            t[k] += u.get(k, 0) or 0
        f = 1.0 if peak(r.get("ts", 0)) else 0.5
        t["usd_micro"] += round(1e6 * f * ((u.get("input_tokens", 0) or 0) * DPRICE[0] + (u.get("cache_read_input_tokens", 0) or 0) * DPRICE[1]
                                           + (u.get("output_tokens", 0) or 0) * DPRICE[2]))
        t["forced"] += bool(r.get("forced"))
        t["max_step"] = max(t["max_step"], r.get("step") or 0)
    return tok


def cost(t):
    """actual deepseek-flash price of the run's calls (peak/off-peak by call time)."""
    return t["usd_micro"] / 1e6


def boot(diff, n=10000, seed=0):
    rng = random.Random(seed)
    m = len(diff)
    vals = sorted(sum(diff[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def compare(A, B, ids):
    a = [int(A[i]["correct"]) for i in ids]; b = [int(B[i]["correct"]) for i in ids]
    d = [y - x for x, y in zip(a, b)]
    fixes = sum(1 for x, y in zip(a, b) if y > x); breaks = sum(1 for x, y in zip(a, b) if x > y)
    lo, hi = boot(d) if ids else (0, 0)
    return dict(n=len(ids), acc_a=sum(a) / max(1, len(ids)), acc_b=sum(b) / max(1, len(ids)), correct_a=sum(a), correct_b=sum(b),
                delta_pp=100 * sum(d) / max(1, len(ids)), ci95_pp=[100 * lo, 100 * hi], fixes=fixes, breaks=breaks,
                mcnemar_p=mcnemar(fixes, breaks))


def slice_of(item):
    if "counting" in item["fine_category"]:
        return "counting"
    return {"spatial_physical": "spatial", "temporal": "temporal", "perception": "perception",
            "reasoning_knowledge": "reasoning"}[item["capability_group"]]


def side(R, ids, tok, run, limit=1800):
    w = [min(R[i]["duration_s"], limit) for i in ids]
    T = [tok[(run, i)] for i in ids]
    sc = [R[i]["n_scan_cmds"] for i in ids]
    calls = [c for i in ids for c in R[i]["scan_calls"]]
    first = [c["sec_total"] for c in calls if c.get("encoded")]
    follow = [c["sec_total"] for c in calls if c.get("encoded") is False and c.get("op") != "reps"]
    return dict(wall_min_mean=st.mean(w) / 60, wall_min_median=st.median(w) / 60, wall_hours_total=sum(w) / 3600,
                timeouts=sum(R[i]["status"] == "timeout" for i in ids), errors=sum(R[i]["status"] == "error" for i in ids),
                tool_calls_mean=st.mean(R[i]["n_tool_calls"] for i in ids),
                frame_steps_mean=st.mean(R[i]["n_frame_steps"] for i in ids), frame_steps_median=st.median(R[i]["n_frame_steps"] for i in ids),
                planner_requests_mean=st.mean(t["requests"] for t in T), planner_requests_median=st.median(t["requests"] for t in T),
                forced_answers=sum(t["forced"] > 0 for t in T),
                warm_s_median=st.median([R[i]["warm"]["sec"] for i in ids if (R[i].get("warm") or {}).get("sec")] or [0]),
                input_tokens_mean=st.mean(t["input_tokens"] + t["cache_read_input_tokens"] for t in T),
                output_tokens_mean=st.mean(t["output_tokens"] for t in T),
                cost_usd_total=sum(cost(t) for t in T), cost_usd_mean=st.mean(cost(t) for t in T),
                scan_cmds_mean=st.mean(sc), items_using_scan=sum(x > 0 for x in sc),
                scan_ops=dict(sum((collections.Counter(R[i]["scan_ops"]) for i in ids), collections.Counter())),
                scan_first_call_s_median=st.median(first) if first else None, scan_followup_s_median=st.median(follow) if follow else None,
                n_scan_calls_logged=len(calls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--ref", default="", help="an earlier run of A's config (run-to-run reference)")
    ap.add_argument("--limit", type=float, default=1800, help="item time limit (s)")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--out", default=""); ap.add_argument("--md", default="")
    a = ap.parse_args()
    I = {json.loads(l)["item_id"]: json.loads(l) for l in open(ITEMS)}
    A, B = load(a.a, a.model), load(a.b, a.model)
    ids = sorted(set(A) & set(B))
    tok = tokens({a.a, a.b, a.ref})
    res = dict(a=a.a, b=a.b, n_a=len(A), n_b=len(B), n_common=len(ids), overall=compare(A, B, ids),
               by_slice={s: compare(A, B, [i for i in ids if slice_of(I[i]) == s]) for s in ("counting", "temporal", "perception", "spatial", "reasoning")},
               by_group={g: compare(A, B, [i for i in ids if I[i]["capability_group"] == g]) for g in sorted({I[i]["capability_group"] for i in ids})},
               by_source={s: compare(A, B, [i for i in ids if I[i]["benchmark"] == s]) for s in sorted({I[i]["benchmark"] for i in ids})},
               cost_time={"a": side(A, ids, tok, a.a, a.limit), "b": side(B, ids, tok, a.b, a.limit)} if ids else {},
               per_item=[dict(item=i, slice=slice_of(I[i]), key=A[i]["expected"], a=A[i]["prediction"], b=B[i]["prediction"],
                              a_ok=A[i]["correct"], b_ok=B[i]["correct"], a_status=A[i]["status"], b_status=B[i]["status"],
                              a_min=round(A[i]["duration_s"] / 60, 1), b_min=round(B[i]["duration_s"] / 60, 1),
                              b_scan=B[i]["n_scan_cmds"], b_ops=B[i]["scan_ops"], a_frames=A[i]["n_frame_steps"], b_frames=B[i]["n_frame_steps"])
                         for i in ids])
    if a.ref:
        Rf = load(a.ref, a.model)
        c = sorted(set(Rf) & set(A))
        res["ref_vs_a"] = dict(ref=a.ref, **compare(Rf, A, c))
        c3 = sorted(set(Rf) & set(B))
        res["ref_vs_b"] = dict(ref=a.ref, **compare(Rf, B, c3))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    o = res["overall"]
    print(f"n={o['n']}  A {o['correct_a']}/{o['n']} ({100*o['acc_a']:.1f}%)  B {o['correct_b']}/{o['n']} ({100*o['acc_b']:.1f}%)  "
          f"delta {o['delta_pp']:+.1f} pp [{o['ci95_pp'][0]:+.1f}, {o['ci95_pp'][1]:+.1f}]  fixes {o['fixes']} breaks {o['breaks']}  McNemar p={o['mcnemar_p']:.3f}")
    for s, v in res["by_slice"].items():
        if v["n"]:
            print(f"  {s:11s} n={v['n']:3d}  {v['correct_a']}->{v['correct_b']}  {v['delta_pp']:+.1f} pp [{v['ci95_pp'][0]:+.0f},{v['ci95_pp'][1]:+.0f}]")
    for k in ("a", "b"):
        if res["cost_time"]:
            print(k, json.dumps({x: (round(y, 2) if isinstance(y, float) else y) for x, y in res["cost_time"][k].items()}))
    if a.ref:
        print("ref vs A:", json.dumps({k: v for k, v in res["ref_vs_a"].items() if k != "ref"}))


if __name__ == "__main__":
    main()
