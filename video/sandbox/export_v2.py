#!/usr/bin/env python3
"""Export a v2 run for sharing (git / HF): per-item result.json + vdmscan call log + per-item planner usage from the
gateway log. Never exports traces (stdout.jsonl holds base64 video frames), prompts or scratch files.
  python3 export_v2.py OUTDIR RUN [RUN ...]"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paired as P  # noqa: E402

out, runs = sys.argv[1], sys.argv[2:]
os.makedirs(out, exist_ok=True)
tok = P.tokens(set(runs))
def sub_usage(cdir):
    """last Claude-subscription utilisation seen by the run (rate_limit_event: five_hour / seven_day fractions)."""
    last = None
    try:
        for line in open(os.path.join(cdir, "stdout.jsonl"), errors="ignore"):
            if '"rate_limit_event"' in line:
                try:
                    last = json.loads(line)["rate_limit_info"].get("unifiedWindows")
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    return last


for run in runs:
    model = json.load(open(os.path.join(P.RUNS, run, "meta.json"))).get("model", "deepseek-v4.1-flash")
    d = os.path.join(P.RUNS, run, model.replace(":", "_"))
    rows = []
    for iid in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        p = os.path.join(d, iid, "result.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        st = P.trace_stats(os.path.join(d, iid))
        t = tok[(run, iid)]
        r.update(n_frame_steps=st["n_frame_steps"], n_scan_cmds=st["n_scan_cmds"], scan_ops=st["scan_ops"], scan_calls=st["scan_calls"],
                 planner_requests=t["requests"], tokens_in_uncached=t["input_tokens"], tokens_in_cached=t["cache_read_input_tokens"],
                 tokens_out=t["output_tokens"], cost_usd=round(P.cost(t), 5), forced_answer=t["forced"] > 0, max_step=t["max_step"], subscription_windows=sub_usage(os.path.join(d, iid)))
        rows.append(r)
    with open(os.path.join(out, f"{run}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    meta = os.path.join(P.RUNS, run, "meta.json")
    if os.path.exists(meta):
        json.dump(json.load(open(meta)), open(os.path.join(out, f"{run}.meta.json"), "w"), indent=1)
    print(run, len(rows))
