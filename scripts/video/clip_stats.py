"""Per-clip motion statistics for the Glance-Video self-supervised generators (CPU, gen-host).

For every normalised clip (8 fps, short side 256) listed in clips/<name>/index.jsonl: decode at 4 fps, 160 px wide,
grayscale; for each consecutive pair compute dense Farneback flow and fit a similarity motion
  u = s*x - r*y + tx,  v = r*x + s*y + ty      (x, y centred, in units of the frame width)
-> per step: tx, ty (fraction of width per step), s (scale rate ~ zoom/forward motion), r (rotation),
   res (median residual flow after removing the global fit = independent object motion, px), diff (mean |I_t - I_t-1|).
Steps are at 0.25 s. Output: stats/<name>.jsonl, one line per clip: {"path", "n", "tx": [...], ...} (rounded).

  python clip_stats.py --root ~/gv --name k700val --workers 4
"""
import argparse, json, os, subprocess
from concurrent.futures import ProcessPoolExecutor
import numpy as np

FPS, WID = 4, 160


def frames(path):
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-i", path, "-an",
           "-vf", f"fps={FPS},scale={WID}:-2:flags=area,format=gray", "-f", "rawvideo", "-"]
    p = subprocess.run(cmd, capture_output=True, timeout=600)
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                         "-of", "csv=p=0", path], capture_output=True, text=True, timeout=60).stdout.strip().split(",")
    W, H = int(pr[0]), int(pr[1])
    h = int(round(H * WID / W / 2)) * 2
    n = len(p.stdout) // (WID * h)
    return np.frombuffer(p.stdout[: n * WID * h], np.uint8).reshape(n, h, WID)


def fit(flow):
    h, w = flow.shape[:2]
    ys, xs = np.mgrid[0:h:4, 0:w:4]
    u = flow[::4, ::4, 0].ravel() / w
    v = flow[::4, ::4, 1].ravel() / w
    x = (xs.ravel() - w / 2) / w
    y = (ys.ravel() - h / 2) / w
    # rows for u and v: unknowns [s, r, tx, ty]
    A = np.zeros((2 * len(x), 4), np.float32)
    A[: len(x)] = np.stack([x, -y, np.ones_like(x), np.zeros_like(x)], 1)
    A[len(x):] = np.stack([y, x, np.zeros_like(x), np.ones_like(x)], 1)
    b = np.concatenate([u, v])
    sol = None
    keep = np.ones(len(b), bool)
    for _ in range(2):  # one re-weighting pass: drop the 30% worst (independently moving objects)
        sol, *_ = np.linalg.lstsq(A[keep], b[keep], rcond=None)
        r = np.abs(A @ sol - b)
        keep = r <= np.quantile(r, 0.7)
    res = np.abs(A @ sol - b)
    return sol, float(np.median(res[: len(x)] + res[len(x):]) * w)


def stats(path):
    import cv2
    try:
        f = frames(path)
    except Exception:
        return path, None
    out = dict(tx=[], ty=[], s=[], r=[], res=[], diff=[])
    for a, b in zip(f[:-1], f[1:]):
        fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        (s, r, tx, ty), res = fit(fl)
        out["tx"].append(round(float(tx), 4)); out["ty"].append(round(float(ty), 4))
        out["s"].append(round(float(s), 4)); out["r"].append(round(float(r), 4))
        out["res"].append(round(res, 3)); out["diff"].append(round(float(np.abs(b.astype(np.int16) - a).mean()), 2))
    return path, dict(n=len(f), **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/gv"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    import cv2
    cv2.setNumThreads(1)
    idx = [json.loads(l) for l in open(f"{a.root}/clips/{a.name}/index.jsonl")]
    os.makedirs(f"{a.root}/stats", exist_ok=True)
    out = f"{a.root}/stats/{a.name}.jsonl"
    done = {json.loads(l)["path"] for l in open(out)} if os.path.exists(out) else set()
    todo = [c["path"] for c in idx if c["path"] not in done]
    print(len(todo), "to do", len(done), "done", flush=True)
    with open(out, "a") as fo, ProcessPoolExecutor(a.workers) as ex:
        for i, (p, st) in enumerate(ex.map(stats, todo, chunksize=8)):
            if st:
                fo.write(json.dumps(dict(path=p, **st)) + "\n")
            if i % 1000 == 0:
                fo.flush(); print(i, flush=True)


if __name__ == "__main__":
    main()
