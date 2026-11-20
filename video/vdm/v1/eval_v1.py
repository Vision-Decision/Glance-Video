#!/usr/bin/env python3
"""Offline evaluation of vdm v1 (clip-level) on the Colab GPU. Settings come from prereg.json (fixed before running).

  python eval_v1.py selftest                 # multi-image wiring: a 1-frame clip must match Glance.ask
  python eval_v1.py count  [--items a,b]     # pre-registered count items (v1.1 rerun + v1.0 held-out)
  python eval_v1.py choose [--n 100]         # choose-only on the held-out v1.0 slice: v0-frame vs glance-clip vs qwen-clip
Videos are fetched from the HF datasets on this VM only (evaluation-only use). Results -> results_v1/*.jsonl
"""
import argparse, json, os, random, subprocess, sys, time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PRE = json.load(open(os.path.join(HERE, "prereg.json")))
CL = PRE["clip"]
OUT = os.path.join(HERE, "results_v1")
os.makedirs(OUT, exist_ok=True)
REPOS = {"v1_0": "Video-Index/Video-Index", "v1_1": "Video-Index/Video-Index-v1-1"}


def items(which):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(REPOS[which], "items/test.jsonl", repo_type="dataset")
    return {json.loads(l)["item_id"]: json.loads(l) for l in open(p)}


def video_path(which, item):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPOS[which], item["video"], repo_type="dataset")


def probe(path):
    o = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height,r_frame_rate:format=duration", "-of", "json", path],
                                  capture_output=True, text=True).stdout)
    s = o["streams"][0]
    n, d = s["r_frame_rate"].split("/")
    return float(o["format"]["duration"]), int(s["width"]), int(s["height"]), float(n) / float(d)


def decode(path, fps):
    """All frames at `fps` (or native if lower), short side CL['frame_short_side']. -> (times, [PIL])"""
    dur, W, H, nfps = probe(path)
    fps = min(fps, nfps)
    ss = CL["frame_short_side"]
    w, h = (ss, int(round(H * ss / W / 2)) * 2) if W <= H else (int(round(W * ss / H / 2)) * 2, ss)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    n = len(raw) // (w * h * 3)
    arr = np.frombuffer(raw[: n * w * h * 3], np.uint8).reshape(n, h, w, 3)
    return [i / fps for i in range(n)], [Image.fromarray(a) for a in arr], fps


def clips_from(times, frames, fps):
    k = CL["frames_per_clip"]
    step = max(1, int(round(CL["stride_seconds"] * fps)))
    cs, ts = [], []
    for s in range(0, max(1, len(frames) - k + 1), step):
        cs.append(frames[s:s + k])
        ts.append(float(np.mean(times[s:s + k])))
    return ts, cs


def cmd_selftest(S):
    from glance import YesNo
    img = Image.new("RGB", (640, 480), "white")
    from PIL import ImageDraw
    ImageDraw.Draw(img).ellipse([200, 140, 440, 380], fill="red")
    q = "Is there a red circle in the image?"
    ref = S.g.ask(img, [YesNo(q)], max_vis_tokens=S.mvt)[0]
    mine = S.score([[img]], q, "glance")[0]
    print("selftest glance 1-frame clip vs Glance.ask:", round(mine, 4), ref)
    for be in ("glance", "qwen"):
        pr = S.score([[img] * 4, [Image.new("RGB", (640, 480), "white")] * 4], q, be)
        print(be, "4-frame red circle vs blank:", [round(p, 3) for p in pr])
        ch = S.score([[img] * 4], "What shape is shown?", be, "choice", ["a square", "a circle", "a star"])
        print(be, "choice square/circle/star:", [round(p, 3) for p in ch[0]])


def cmd_count(S, only):
    from core import count_distribution, segments
    groups = PRE["count_items"]
    I = {"v1_1": items("v1_1"), "v1_0": items("v1_0")}
    out = open(os.path.join(OUT, "count.jsonl"), "a")
    for gname, which in (("v1.1 (rerun, compare with v0 6/64/27/16)", "v1_1"),
                         ("v1.0 held-out (action_counting, not in v1.1)", "v1_0")):
        for iid, spec in groups[gname].items():
            if only and iid not in only:
                continue
            t0 = time.time()
            times, frames, fps = decode(video_path(which, I[which][iid]), CL["decode_fps"])
            ts, cs = clips_from(times, frames, fps)
            rec = {"item_id": iid, "set": which, "key": spec["key"], "query": spec["q"], "n_clips": len(cs),
                   "fps": fps, "decode_s": round(time.time() - t0, 1)}
            for be in ("glance", "qwen"):
                t1 = time.time()
                ps = S.score(cs, spec["q"], be)
                dt = time.time() - t1
                pr = PRE["count_rule"]["primary"]
                c = len(segments(ts, ps, pr["hi"], pr["lo"], pr["merge_gap_s"]))
                rec[be] = {"count": c, "exact": c == spec["key"], "within1": abs(c - spec["key"]) <= 1,
                           "dist": count_distribution(ts, ps), "clips_per_s": round(len(cs) / dt, 2),
                           "curve": [[round(t, 1), round(p, 3)] for t, p in zip(ts, ps)]}
            out.write(json.dumps(rec) + "\n"); out.flush()
            print(iid, "key", spec["key"], "glance", rec["glance"]["count"], "qwen", rec["qwen"]["count"],
                  f"clips {len(cs)} @ {rec['glance']['clips_per_s']}/s", flush=True)


def cmd_choose(S, n):
    I0 = items("v1_0")
    v11 = set(items("v1_1"))
    held = sorted(i for i in I0 if i not in v11)
    sl = random.Random(0).sample(held, n)
    fn = os.path.join(OUT, "choose.jsonl")
    done = {json.loads(l)["item_id"] for l in open(fn)} if os.path.exists(fn) else set()
    out = open(fn, "a")
    k = CL["frames_per_clip"]
    for iid in sl:
        if iid in done:
            continue
        x = I0[iid]
        t0 = time.time()
        times, frames, fps = decode(video_path("v1_0", x), CL["decode_fps"])
        N = len(frames)
        single = [[frames[int(i)]] for i in np.linspace(0, N - 1, 32)]
        starts = [int(s) for s in np.linspace(0, max(0, N - k), 8)]
        clips = [frames[s:s + k] if N >= k else frames + [frames[-1]] * (k - N) for s in starts]
        rec = {"item_id": iid, "group": x["capability_group"], "benchmark": x["benchmark"], "key": x["answer_idx"],
               "n_options": len(x["options"])}
        for name, cs, be, mvt in (("v0-frame", single, "glance", 640), ("glance-clip", clips, "glance", None),
                                  ("qwen-clip", clips, "qwen", None)):
            old = S.mvt
            S.mvt = mvt or old
            P = np.array(S.score(cs, x["question"], be, "choice", x["options"], batch=8))
            S.mvt = old
            m = P.mean(0)
            rec[name] = {"pred": int(m.argmax()), "argmin": int(m.argmin()), "probs": [round(float(v), 4) for v in m]}
        rec["sec"] = round(time.time() - t0, 1)
        out.write(json.dumps(rec) + "\n"); out.flush()
        print(iid, x["capability_group"], "key", x["answer_idx"], {k2: rec[k2]["pred"] for k2 in ("v0-frame", "glance-clip", "qwen-clip")}, rec["sec"], "s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "count", "choose"])
    ap.add_argument("--items", default="")
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    from core import ClipScorer
    S = ClipScorer(max_vis_tokens=CL["max_vis_tokens_per_frame"])
    {"selftest": lambda: cmd_selftest(S), "count": lambda: cmd_count(S, set(filter(None, a.items.split(",")))),
     "choose": lambda: cmd_choose(S, a.n)}[a.cmd]()


if __name__ == "__main__":
    main()
