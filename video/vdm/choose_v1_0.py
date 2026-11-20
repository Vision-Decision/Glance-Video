#!/usr/bin/env python3
"""vdm v0 choose-only over Video-Index v1.0 (840 items, normalized videos). Same settings as choose_all.py
(32 frames, agent image, 2 CPUs, same frozen v0 client vdm/vdm). Resumable; at most --par containers at once.
Infrastructure failures (vdm server unreachable / 5xx) are NOT scored: the item is retried with backoff for up to
~15 min, then written to <out>.errors.jsonl and left for the next invocation (only scored records count as done)."""
import argparse, json, os, subprocess, threading, time, concurrent.futures as cf
H = "~/videoindex"
ap = argparse.ArgumentParser()
ap.add_argument("--par", type=int, default=3)
ap.add_argument("--out", default=f"{H}/vdm/results/choose_only_v1_0.jsonl")
a = ap.parse_args()
items = [json.loads(l) for l in open(f"{H}/data/v1_0/items/test.jsonl")]
done = {r["item_id"] for r in map(json.loads, open(a.out)) if r.get("pred")} if os.path.exists(a.out) else set()
lock = threading.Lock()

def one(x):
    opts = [f"{chr(65 + i)}. {o}" for i, o in enumerate(x["options"])]
    name = "vdmchoose10_" + x["item_id"].replace("^", "_")
    cmd = ["docker", "run", "--rm", "--name", name, "--cpus=2", "--network", "bridge", "-e", "VDM_URL=http://172.17.0.1:18080",
           "-e", "VDM_CACHE=/tmp/vdm_cache", "-v", f"{H}/data/v1_0/{x['video']}:/work/video.mp4:ro",
           "-v", f"{H}/vdm/vdm:/usr/local/bin/vdm:ro", "-w", "/work", "vi-agent:cc2.1.300",
           "vdm", "choose", "video.mp4", "--question", x["question"], "--frames", "32", "--options", *opts]
    t = time.time()
    rec = {"item_id": x["item_id"], "benchmark": x["benchmark"], "group": x["capability_group"],
           "key": x["answer"], "n_options": len(opts)}
    wait, err = 20, ""
    for attempt in range(8):
        r = None
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            o = json.loads(r.stdout.splitlines()[0])
            if o.get("answer"):
                rec.update(pred=o["answer"], mean_prob=o["mean_prob"], correct=o["answer"] == x["answer"])
                break
            err = json.dumps(o)[:300]
        except Exception as e:
            err = ((r.stderr or r.stdout)[-300:] if r is not None else f"{type(e).__name__}")
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        time.sleep(wait)
        wait = min(wait * 2, 240)
    rec["sec"] = round(time.time() - t, 1)
    with lock, open(a.out if rec.get("pred") else a.out + ".errors.jsonl", "a") as f:
        f.write(json.dumps(rec if rec.get("pred") else {**rec, "error": err}) + "\n")

import random
todo = sorted((x for x in items if x["item_id"] not in done), key=lambda x: x["item_id"])
random.Random(0).shuffle(todo)  # seeded random order: any partial result is a random subset
print("todo", len(todo), flush=True)
with cf.ThreadPoolExecutor(a.par) as ex:
    list(ex.map(one, todo))
recs = [r for r in map(json.loads, open(a.out)) if r.get("pred")]
print(f"choose-only v1.0: {sum(r['correct'] for r in recs)}/{len(recs)} scored; {len(items) - len(recs)} unscored", flush=True)
