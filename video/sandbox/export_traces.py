#!/usr/bin/env python3
"""Export a run's traces in the official redacted form (run on the host that holds the run).

  python3 export_traces.py RUNS_DIR RUN OUT_DIR
Copies prompt.txt, command.json, schema.json, result.json, stderr.txt and stdout.jsonl of every item; in
stdout.jsonl, base64 image data is replaced by {"sha256": ..., "bytes": ...} like the published traces.
Extracted frames/audio (work/) are not copied. Also copies per_question.csv / summary.json / meta.json and the
gateway usage records of this run.
"""
import base64, hashlib, json, os, shutil, sys


def redact(x):
    if isinstance(x, dict):
        if x.get("type") == "base64" and isinstance(x.get("data"), str) and len(x["data"]) > 200:
            raw = base64.b64decode(x["data"] + "===")
            return {"type": "redacted_image", "media_type": x.get("media_type"),
                    "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        return {k: redact(v) for k, v in x.items()}
    if isinstance(x, list):
        return [redact(v) for v in x]
    return x


def main(runs, run, out):
    src = os.path.join(runs, run)
    os.makedirs(out, exist_ok=True)
    for f in ("per_question.csv", "summary.json", "meta.json"):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy(os.path.join(src, f), out)
    for model in os.listdir(src):
        md = os.path.join(src, model)
        if not os.path.isdir(md):
            continue
        for iid in os.listdir(md):
            s, d = os.path.join(md, iid), os.path.join(out, model, iid)
            os.makedirs(d, exist_ok=True)
            for f in ("prompt.txt", "command.json", "schema.json", "result.json", "stderr.txt"):
                if os.path.exists(os.path.join(s, f)):
                    shutil.copy(os.path.join(s, f), d)
            if os.path.exists(os.path.join(s, "stdout.jsonl")):
                with open(os.path.join(s, "stdout.jsonl"), errors="ignore") as fi, \
                        open(os.path.join(d, "stdout.jsonl"), "w") as fo:
                    for line in fi:
                        try:
                            ev = json.loads(line)
                        except Exception:
                            continue
                        if ev.get("type") == "system" and ev.get("subtype") == "thinking_tokens":
                            continue  # progress ticks, thousands per item
                        fo.write(json.dumps(redact(ev)) + "\n")
    g = os.path.join(runs, "_gateway", "usage.jsonl")
    if os.path.exists(g):
        with open(g) as fi, open(os.path.join(out, "gateway_usage.jsonl"), "w") as fo:
            for line in fi:
                if json.loads(line)["call"].startswith(run + "."):
                    fo.write(line)


if __name__ == "__main__":
    main(*sys.argv[1:4])
