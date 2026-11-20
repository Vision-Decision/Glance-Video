#!/usr/bin/env python3
"""'vdm choose-only' lower bound: answer every v1.1 item with `vdm choose` alone (no LLM, no audio).

  python3 choose_all.py [--frames 32] [--out results/choose_only.jsonl]   (run on gen-host; resumable)
Each item runs in the agent image (2 CPUs) with the video read-only, exactly like the agent would call the tool.
"""
import argparse, json, os, subprocess, time

HOME = "~/videoindex"
ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=32)
ap.add_argument("--out", default=f"{HOME}/vdm/results/choose_only.jsonl")
ap.add_argument("--url", default="http://172.17.0.1:18080")
a = ap.parse_args()
items = [json.loads(l) for l in open(f"{HOME}/research/data/test.jsonl")]
done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["item_id"] for l in open(a.out)}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
for x in items:
    if x["item_id"] in done:
        continue
    opts = [f"{chr(65 + i)}. {o}" for i, o in enumerate(x["options"])]
    cmd = ["docker", "run", "--rm", "--cpus=2", "--network", "bridge", "-e", f"VDM_URL={a.url}",
           "-e", "VDM_CACHE=/tmp/vdm_cache", "-v", f"{HOME}/data/v1_1/{x['video']}:/work/video.mp4:ro",
           "-v", f"{HOME}/vdm/vdm:/usr/local/bin/vdm:ro", "-w", "/work", "vi-agent:cc2.1.300",
           "vdm", "choose", "video.mp4", "--question", x["question"], "--frames", str(a.frames), "--options", *opts]
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    rec = {"item_id": x["item_id"], "benchmark": x["benchmark"], "group": x["capability_group"],
           "key": x["answer"], "n_options": len(opts), "sec": round(time.time() - t, 1)}
    try:
        o = json.loads(r.stdout.splitlines()[0])
        rec.update(pred=o["answer"], mean_prob=o["mean_prob"], correct=o["answer"] == x["answer"])
    except Exception:
        rec.update(pred=None, correct=False, error=(r.stderr or r.stdout)[-300:])
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(rec["item_id"], rec.get("pred"), rec["key"], rec["sec"], flush=True)
recs = [json.loads(l) for l in open(a.out)]
print(f"choose-only: {sum(r['correct'] for r in recs)}/{len(recs)} = {100*sum(r['correct'] for r in recs)/len(recs):.1f}%")
