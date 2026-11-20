#!/usr/bin/env python3
"""Find finished +vdm items whose agent saw a vdm *infrastructure* error, and (with --move) take them out of scoring.

  python3 sandbox/flag_vdm_errors.py RUN [RUN ...] [--move]
An infrastructure error is what the vdm client prints when no server answered: "server error 50x",
"vdm server unreachable", "vdm server unavailable ... after retrying", "vdm temporarily unavailable". Usage errors
("empty range", bad arguments) do not count. With --move the item dir goes to runs/_compromised/<run>_<model>_<item>
with a compromised.json (reason, matched messages), so the next invocation of run.py re-runs it from scratch.
"""
import argparse, datetime as dt, glob, json, os, re, shutil

RUNS = os.path.join(os.environ.get("VI_DATA", "~/videoindex/data"), "runs")
PAT = re.compile(r"server error 50[234]|vdm server unreachable|vdm server unavailable|vdm temporarily unavailable")


def strings(x):
    if isinstance(x, dict):
        for k, v in x.items():
            if k in ("data", "source", "inlineData") and len(str(v)) > 5000:
                continue
            yield from strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from strings(v)
    elif isinstance(x, str) and len(x) < 200000:
        yield x


def tool_outputs(f):
    """Strings that are tool *results* (what the agent saw), not the agent's own text."""
    for line in open(f, errors="ignore"):
        if "tool_result" not in line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") == "tool_result":                      # Gemini CLI stream-json
            yield from strings(ev.get("output"))
        elif ev.get("type") == "user":                           # Claude Code stream-json
            for c in (ev.get("message") or {}).get("content", []):
                if isinstance(c, dict) and c.get("type") == "tool_result":
                    yield from strings(c.get("content"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--move", action="store_true")
    a = ap.parse_args()
    for run in a.runs:
        for d in sorted(glob.glob(os.path.join(RUNS, run, "*", "*"))):
            if not os.path.exists(os.path.join(d, "result.json")):
                continue
            hits = [m.group(0) for s in tool_outputs(os.path.join(d, "stdout.jsonl")) for m in PAT.finditer(s)]
            if not hits:
                continue
            model, iid = d.split(os.sep)[-2:]
            print(f"{run}/{iid}: {len(hits)} vdm infra errors {sorted(set(hits))}")
            if a.move:
                dst = os.path.join(RUNS, "_compromised", f"{run}_{model}_{iid}")
                json.dump({"run": run, "item": iid, "reason": "vdm infrastructure error seen by the agent",
                           "messages": sorted(set(hits)), "n": len(hits),
                           "flagged_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                           "result_before": json.load(open(os.path.join(d, "result.json")))},
                          open(os.path.join(d, "compromised.json"), "w"), indent=1)
                os.remove(os.path.join(d, "result.json"))
                if os.path.exists(os.path.join(d, "claimed")):
                    os.remove(os.path.join(d, "claimed"))
                shutil.move(d, dst if not os.path.exists(dst) else dst + "_" + dt.datetime.now().strftime("%H%M%S"))


if __name__ == "__main__":
    main()
