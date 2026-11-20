#!/usr/bin/env python3
"""Claude-subscription usage of Opus runs: utilisation of the 5-hour and 7-day windows as reported by Claude Code's
rate_limit_event, first/last value per item, in time order.  python3 opus_usage.py RUN [RUN ...]"""
import glob, json, os, sys
RUNS = os.path.join(os.environ.get("VI_DATA", "."), "runs")
rows = []
for run in sys.argv[1:]:
    for f in glob.glob(os.path.join(RUNS, run, "claude-opus-5-5", "*", "stdout.jsonl")):
        ev = []
        for line in open(f, errors="ignore"):
            if '"rate_limit_event"' in line:
                try:
                    w = json.loads(line)["rate_limit_info"]["unifiedWindows"]
                    ev.append((w["five_hour"]["utilization"], w["seven_day"]["utilization"], w["seven_day"]["resetsAt"]))
                except Exception:
                    pass
        if ev:
            done = os.path.exists(os.path.join(os.path.dirname(f), "result.json"))
            rows.append((os.path.getmtime(f), run, f.split("/")[-2], ev[0], ev[-1], done))
for mt, run, iid, a, b, done in sorted(rows):
    print(f"{run:18s} {iid:24s} 5h {a[0]:.2f}->{b[0]:.2f}  7d {a[1]:.2f}->{b[1]:.2f}  {'done' if done else 'running'}")
if rows:
    s = sorted(rows)
    print(f"7-day window: first seen {s[0][3][1]:.2f}, last {s[-1][4][1]:.2f}; finished items {sum(r[5] for r in rows)}; 7d resets at {s[-1][4][2]}")
