"""RepCount-A (LLSP) from the HF Lance mirror lmms-lab-eval/repcounta-lance -> normalised clips + index (gen-host).

  python prep_repcount.py --root ~/gv --workers 2
-> clips/repcount_{train,validation,test}/<video_id>.mp4 (8 fps, short side 256) and index.jsonl rows:
   {path, rel, id, split, action, count, cycles: [[start_s, end_s], ...] (from the L1..L302 frame markers), fps_src, n, w, h, dur}
Cycle markers are frame indices of the original video (pairs: start, end of each repetition).
"""
import argparse, json, os, subprocess, sys, tempfile
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_clips import normalise, probe


def src_fps(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                        "-of", "csv=p=0", p], capture_output=True, text=True).stdout.strip()
    n, d = (r.split("/") + ["1"])[:2]
    return float(n) / float(d or 1)


def job(x):
    raw, dst, meta = x
    fps = src_fps(raw)
    d, info = normalise((raw, dst, None, None))
    if not info:
        return None
    b = [float(v) for v in meta["bounds"] if v is not None and str(v) != "nan"]
    cyc = [[round(b[i] / fps, 3), round(b[i + 1] / fps, 3)] for i in range(0, len(b) - 1, 2)]
    return dict(path=dst, id=meta["id"], split=meta["split"], action=meta["action"], count=meta["count"], cycles=cyc,
                fps_src=fps, **info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/gv"))
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    import lance
    base = os.path.dirname(a.root.rstrip("/"))
    rawdir = f"{a.root}/raw/repcount_mp4"
    os.makedirs(rawdir, exist_ok=True)
    for split in ("train", "validation", "test"):
        ds = lance.dataset(f"{a.root}/raw/repcount_lance/data/{split}.lance")
        out = f"{a.root}/clips/repcount_{split}"
        os.makedirs(out, exist_ok=True)
        jobs = []
        cols = ["video_id", "split", "action_type", "count", "cycle_bounds_json", "video_ext"]
        meta = ds.to_table(columns=cols).to_pylist()
        for i, r in enumerate(meta):
                if r["count"] is None:
                    continue
                raw = f"{rawdir}/{r['video_id']}.{(r['video_ext'] or 'mp4').lstrip('.')}"
                if not os.path.exists(raw):
                    try:
                        bf = ds.take_blobs("video_blob", indices=[i])[0]
                    except TypeError:
                        bf = ds.take_blobs(blob_column="video_blob", indices=[i])[0]
                    with bf as fh:
                        open(raw, "wb").write(fh.read())
                jobs.append((raw, f"{out}/{r['video_id']}.mp4",
                             dict(id=r["video_id"], split=split, action=r["action_type"], count=int(r["count"]),
                                  bounds=json.loads(r["cycle_bounds_json"] or "[]"))))
        with ProcessPoolExecutor(a.workers) as ex, open(f"{out}/index.jsonl", "w") as f:
            for row in ex.map(job, jobs):
                if row:
                    row["rel"] = os.path.relpath(row["path"], base)
                    f.write(json.dumps(row) + "\n")
        print(split, len(jobs), flush=True)


if __name__ == "__main__":
    main()
