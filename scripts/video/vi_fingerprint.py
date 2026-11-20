"""Fingerprint every Video-Index video (v1.0 + v1.1) with per-frame pHashes -> npz (hash, owner=video_id).
Run on gen-host: python vi_fingerprint.py --out ~/gv/decon/vi_hashes.npz --workers 4"""
import argparse, glob, json, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from vhash import frame_hashes


def dur(p):
    try:
        return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                                    capture_output=True, text=True, timeout=120).stdout.strip())
    except Exception:
        return 0.0


def job(p):
    d = dur(p)
    fps = 2.0 if d <= 600 else (0.5 if d <= 3600 else 0.25)
    try:
        h = frame_hashes(p, fps)
    except Exception as e:
        print("ERR", p, e, flush=True)
        h = np.zeros(0, np.uint64)
    return os.path.basename(p).split(".")[0], d, fps, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=[os.path.expanduser("~/videoindex/data/v1_0/videos"),
                                                   os.path.expanduser("~/videoindex/data/v1_1/videos")])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    paths = sorted(set(p for d in a.dirs for p in glob.glob(f"{d}/*.mp4")))
    print(len(paths), "videos", flush=True)
    H, O, meta = [], [], {}
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (vid, d, fps, h) in enumerate(ex.map(job, paths)):
            H.append(h); O += [vid] * len(h); meta[vid] = dict(duration=d, fps=fps, n=len(h))
            if i % 50 == 0:
                print(i, vid[:12], d, len(h), flush=True)
    np.savez_compressed(a.out, hash=np.concatenate(H) if H else np.zeros(0, np.uint64), owner=np.array(O))
    json.dump(meta, open(a.out.replace(".npz", "_meta.json"), "w"))
    print("DONE", sum(len(h) for h in H), "frames", flush=True)


if __name__ == "__main__":
    main()
