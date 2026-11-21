"""Decode every source video ONCE and store it as 256x256 H.264 (CRF 17, tune fastdecode, original timestamps) under
DATA/cache256/<path relative to DATA/raw>. vio.decode reads the cached file when it exists: decoding 256x256 costs a
fraction of 480p / 1080p, so every later eval arm and every training epoch skips the expensive decode.
python video08/data/cache.py DATA [workers] [eval|train]   eval: Charades test + PushupBench + RepCount test;
train: Charades train/probe_val + RepCount train/val. Every encoder / decoder is single-threaded (workers = cores used)."""
import glob, json, os, sys, time
from multiprocessing import Pool
import av

R = sys.argv[1]; W = int(sys.argv[2]) if len(sys.argv) > 2 else 12; S = 256


def out_path(p):
    return os.path.join(R, "cache256", os.path.relpath(p, os.path.join(R, "raw")))


def one(p):
    o = out_path(p)
    if os.path.exists(o):
        return 0
    os.makedirs(os.path.dirname(o), exist_ok=True); tmp = o + ".tmp.mp4"; n = 0
    try:
        with av.open(p) as src, av.open(tmp, "w") as dst:
            s = src.streams.video[0]; s.thread_count = 1
            rate = s.average_rate or s.guessed_rate or 30
            v = dst.add_stream("libx264", rate=rate); v.width = v.height = S; v.pix_fmt = "yuv420p"
            v.time_base = s.time_base; v.options = {"crf": "17", "preset": "veryfast", "tune": "fastdecode", "g": "30", "threads": "1"}; v.thread_count = 1
            v.codec_context.time_base = s.time_base
            last = None
            for fr in src.decode(s):
                if fr.pts is None or (last is not None and fr.pts <= last):
                    continue
                f2 = fr.reformat(width=S, height=S, format="yuv420p"); f2.pts = fr.pts; f2.time_base = s.time_base; last = fr.pts
                for pk in v.encode(f2):
                    dst.mux(pk)
                n += 1
            for pk in v.encode():
                dst.mux(pk)
        os.replace(tmp, o)
    except Exception as e:
        print("fail", p, repr(e)[:200], flush=True)
        if os.path.exists(tmp):
            os.remove(tmp)
    return n


def npy(args):
    """uint8 frames on the k/fps grid (decoded from the 256 cache) -> DATA/cache_fps<f>/<rel>.npy + .t.npy"""
    p, fps = args
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from video08 import vio
    import numpy as np
    o = os.path.splitext(p.replace("/raw/", f"/cache_fps{fps:g}/", 1))[0] + ".npy"
    if os.path.exists(o):
        return 0
    os.makedirs(os.path.dirname(o), exist_ok=True)
    try:
        x, t = vio._decode(vio.cached(p), fps, S, 0.0, 180.0, threads=1)
        np.save(o[:-4] + ".tmp.npy", x.numpy()); np.save(o[:-4] + ".t.npy", t); os.replace(o[:-4] + ".tmp.npy", o)
        return len(t)
    except Exception as e:
        print("npy fail", p, repr(e)[:200], flush=True); return 0


if __name__ == "__main__":
    os.nice(10)
    idx = f"{R}/idx"; rows = {r["id"]: r for r in map(json.loads, open(f"{idx}/charades.jsonl"))}
    sp = json.load(open(f"{idx}/splits.json"))
    part = sys.argv[3] if len(sys.argv) > 3 else "eval"
    if part == "eval":
        paths = [rows[i]["path"] for i in sp["eval"]] + sorted(glob.glob(f"{R}/raw/pushup/*.mp4"))
        splits_ = ("test",)
    else:
        paths = [rows[i]["path"] for i in sp["train"] + sp["probe_val"]]
        splits_ = ("validation", "train")
    for s_ in splits_:
        f = f"{idx}/repcount_{s_}.jsonl"
        if os.path.exists(f):
            paths += [json.loads(l)["path"] for l in open(f)]
    t0 = time.time(); tot = 0
    with Pool(W) as pool:
        for i, n in enumerate(pool.imap(one, paths, chunksize=4)):
            tot += n
            if i % 200 == 0:
                print(i, len(paths), tot, "frames", round(tot / max(time.time() - t0, 1)), "fps", flush=True)
    print("done", len(paths), tot, flush=True)
    if part == "eval":  # zero-decode frame cache for the fps-2 evaluation arms (small: ~20 GB)
        with Pool(W) as pool:
            n = sum(pool.imap(npy, [(p, 2.0) for p in paths], chunksize=4))
        print("npy fps2 done", n, flush=True)
