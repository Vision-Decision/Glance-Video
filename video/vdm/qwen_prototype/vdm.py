#!/usr/bin/env python3
"""vdm prototype: scan a video with short clips + yes/no questions -> P(t), intervals, counts.

  vdm.py find  "Is the person opens the fridge?" --video video.mp4 [--clip 2 --stride 1 --frames 4]
  vdm.py count "Is the person opens the fridge?" --video video.mp4 [--start S --end E]

Output: JSON on stdout (P(t) curve, intervals, count) and optionally --png / --json files.
Frames are decoded once with ffmpeg at (frames/clip) fps-equivalent and grouped into overlapping clips, which is
equivalent to cutting each clip with ffmpeg but decodes the video only once.
"""
import argparse
import json
import subprocess
import sys
import time

import numpy as np
from PIL import Image


def probe(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:format=duration", "-of", "json", video],
                         capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    s = d["streams"][0]
    return float(d["format"]["duration"]), int(s["width"]), int(s["height"])


def decode(video, fps, short, start=0.0, end=None):
    """Yield (t, PIL frame) at `fps`, resized so the short side is `short` (multiple of 32)."""
    dur, W, H = probe(video)
    end = min(end or dur, dur)
    if W <= H:
        w, h = short, int(round(H * short / W / 32) * 32)
    else:
        w, h = int(round(W * short / H / 32) * 32), short
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start), "-to", str(end), "-i", video,
           "-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)
    n, i = w * h * 3, 0
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield start + i / fps, Image.frombuffer("RGB", (w, h), b)
        i += 1
    p.wait()


def clips_from_frames(frames, clip, stride, k):
    """frames: list of (t, img) at k/clip fps. Returns list of (t0, t1, [k imgs])."""
    fps = k / clip
    step = max(1, int(round(stride * fps)))
    out = []
    for s in range(0, max(1, len(frames) - k + 1), step):
        grp = frames[s:s + k]
        if len(grp) < k:
            break
        out.append((grp[0][0], grp[0][0] + clip, [im for _, im in grp]))
    return out


def scan(scorer, video, question, clip=2.0, stride=1.0, k=4, short=336, start=0.0, end=None, batch=16, ctx=""):
    """Streams frames: decode -> overlapping clips -> batched scoring, so long movies fit in memory."""
    t0 = time.time()
    step = max(1, int(round(stride * k / clip)))
    buf, base, clips, p, pending, t_gpu = [], 0, [], [], [], 0.0

    def flush():
        nonlocal t_gpu, pending
        if pending:
            t1 = time.time()
            p.extend(scorer.score([c[2] for c in pending], question, clip, ctx))
            t_gpu += time.time() - t1
            clips.extend((c[0], c[1]) for c in pending)
            pending = []

    buf = {}
    for idx, (t, im) in enumerate(decode(video, k / clip, short, start, end)):
        if idx >= base:
            buf[idx] = (t, im)
        # emit every clip whose k frames (base .. base+k-1) have arrived
        while base + k <= idx + 1:
            grp = [buf[j] for j in range(base, base + k)]
            pending.append((grp[0][0], grp[0][0] + clip, [x for _, x in grp]))
            base += step
            for j in [j for j in buf if j < base]:
                del buf[j]
            if len(pending) >= batch:
                flush()
    flush()
    t_all = time.time() - t0
    clips = [(a, b, None) for a, b in clips]
    t_dec = t_all - t_gpu
    return {"question": question, "clip": clip, "stride": stride, "frames_per_clip": k, "short_side": short,
            "start": start, "end": end, "t": [c[0] for c in clips], "t_end": [c[1] for c in clips],
            "p": [round(x, 4) for x in p], "n_clips": len(clips), "decode_s": round(t_dec, 1),
            "score_s": round(t_gpu, 1), "clips_per_s": round(len(clips) / max(t_gpu, 1e-6), 2),
            "wall_s": round(t_all, 1)}


def intervals(res, hi=0.5, lo=None, min_gap=0.0, min_len=0.0):
    """Hysteresis threshold on P(t): enter at >=hi, stay while >=lo; merge gaps < min_gap seconds.
    Each clip stands for [center - stride/2, center + stride/2], so one low clip separates two events."""
    lo = hi * 0.6 if lo is None else lo
    half = res["stride"] / 2
    out, cur = [], None
    for t0, t1, p in zip(res["t"], res["t_end"], res["p"]):
        c = (t0 + t1) / 2
        t, te = c - half, c + half
        if cur is None:
            if p >= hi:
                cur = [t, te, p]
        elif p >= lo:
            cur[1], cur[2] = te, max(cur[2], p)
        else:
            out.append(cur)
            cur = None
    if cur:
        out.append(cur)
    merged = []
    for iv in out:
        if merged and iv[0] - merged[-1][1] < min_gap:
            merged[-1][1], merged[-1][2] = iv[1], max(merged[-1][2], iv[2])
        else:
            merged.append(iv)
    return [{"start": round(a, 2), "end": round(b, 2), "peak": round(c, 3)} for a, b, c in merged
            if b - a >= min_len]


def plot(res, ivs, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array(res["t"]) + res["clip"] / 2
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.plot(t, res["p"], lw=1, color="#3b6ea8")
    for iv in ivs:
        ax.axvspan(iv["start"], iv["end"], color="#e8a33d", alpha=0.3, lw=0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("P(yes)")
    ax.set_title(title or res["question"], fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["find", "count"])
    ap.add_argument("question")
    ap.add_argument("--video", required=True)
    ap.add_argument("--clip", type=float, default=2.0)
    ap.add_argument("--stride", type=float, default=1.0)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--short", type=int, default=336)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--hi", type=float, default=0.5)
    ap.add_argument("--lo", type=float, default=None)
    ap.add_argument("--min-gap", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--png")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    from qwen_backend import QwenYesNo
    scorer = QwenYesNo(a.model)
    res = scan(scorer, a.video, a.question, a.clip, a.stride, a.frames, a.short, a.start, a.end, a.batch)
    ivs = intervals(res, a.hi, a.lo, a.min_gap)
    res["intervals"] = ivs
    if a.cmd == "count":
        res["count"] = len(ivs)
    if a.png:
        plot(res, ivs, a.png)
    if a.json:
        json.dump(res, open(a.json, "w"))
    summary = {k: v for k, v in res.items() if k not in ("t", "t_end", "p")}
    json.dump(summary, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
