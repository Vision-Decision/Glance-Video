"""Glance-Video clip pool on gen-host: normalise source videos to small fixed-rate clips, then decontaminate.

  python prep_clips.py k700 --root ~/gv --workers 6          # Kinetics-700-2020 val tars -> clips/k700val/<class>/<id>.mp4
  python prep_clips.py charades --root ~/gv --workers 6      # Charades_v1_480 -> clips/charades/<id>.mp4
  python prep_clips.py dir --src DIR --name NAME [--seg 60]  # any folder of videos (optionally cut into segments)
  python prep_clips.py decon --root ~/gv --workers 6         # pHash every clip at 1 fps vs Video-Index -> decon/

Normalised clip: 8 fps, short side 256 (even dims), H.264 CRF 26, no audio. Every clip is listed in
clips/<name>/index.jsonl (path, source id, duration, w, h, n_frames).
"""
import argparse, glob, json, os, subprocess, sys, tarfile, zipfile
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FPS, SHORT = 8, 256


def probe(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                        "stream=width,height,nb_read_packets:format=duration", "-of", "json", p],
                       capture_output=True, text=True, timeout=300)
    j = json.loads(r.stdout or "{}")
    s = (j.get("streams") or [{}])[0]
    return dict(w=s.get("width"), h=s.get("height"), n=int(s.get("nb_read_packets") or 0),
                dur=float(j.get("format", {}).get("duration") or 0))


def normalise(job):
    src, dst, ss, t = job
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst, probe(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-threads", "1"]
    if ss is not None:
        cmd += ["-ss", f"{ss:.3f}"]
    cmd += ["-i", src]
    if t is not None:
        cmd += ["-t", f"{t:.3f}"]
    vf = f"fps={FPS},scale='if(lt(iw,ih),{SHORT},-2)':'if(lt(iw,ih),-2,{SHORT})':flags=area,format=yuv420p"
    cmd += ["-an", "-sn", "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-g", "16",
            "-x264-params", "threads=1", dst + ".tmp.mp4"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if r.returncode or not os.path.exists(dst + ".tmp.mp4"):
        return dst, None
    os.replace(dst + ".tmp.mp4", dst)
    return dst, probe(dst)


def run_jobs(jobs, idx_path, workers, meta):
    os.makedirs(os.path.dirname(idx_path), exist_ok=True)
    done = set()
    if os.path.exists(idx_path):
        done = {json.loads(l)["path"] for l in open(idx_path)}
    jobs = [j for j in jobs if j[1] not in done]
    import random
    random.Random(0).shuffle(jobs)  # any prefix of the run is a random subset (classes / videos)
    print(len(jobs), "to do;", len(done), "done", flush=True)
    with open(idx_path, "a") as f, ProcessPoolExecutor(workers) as ex:
        for i, (dst, info) in enumerate(ex.map(normalise, jobs, chunksize=4)):
            if info and info["n"] >= 8:
                f.write(json.dumps(dict(path=dst, **meta(dst), **info)) + "\n")
            if i % 500 == 0:
                f.flush(); print(i, dst, info, flush=True)


def cmd_k700(a):
    raw = f"{a.root}/raw/k700val"
    ex = f"{raw}/x"
    os.makedirs(ex, exist_ok=True)
    for t in sorted(glob.glob(f"{raw}/*.tar.gz")):
        mark = t + ".extracted"
        if not os.path.exists(mark):
            with tarfile.open(t) as tf:
                tf.extractall(ex, filter="data")
            open(mark, "w").close()
    srcs = sorted(glob.glob(f"{ex}/*/*.mp4"))
    out = f"{a.root}/clips/k700val"
    jobs = [(s, f"{out}/{os.path.basename(os.path.dirname(s)).replace(' ', '_')}/{os.path.basename(s)}", None, None) for s in srcs]

    def meta(dst):
        b = os.path.basename(dst)[:-4]
        return dict(src="k700val", label=os.path.basename(os.path.dirname(dst)).replace("_", " "), ytid=b[:11],
                    t0=int(b[12:18]), t1=int(b[19:25]))
    run_jobs(jobs, f"{out}/index.jsonl", a.workers, meta)


def cmd_countix_train(a):
    """only the Countix-train clips out of the per-class Kinetics-700-2020 train tars (raw/k700train/*.tar.gz)"""
    import csv
    raw = f"{a.root}/raw/k700train"
    ex = f"{raw}/x"
    os.makedirs(ex, exist_ok=True)
    want = {}
    for r in csv.DictReader(open(f"{a.root}/raw/ann/countix/countix_train.csv")):
        want[f"{r['video_id']}_{int(r['kinetics_start']):06d}_{int(r['kinetics_end']):06d}.mp4"] = r["class"]
    for t in sorted(glob.glob(f"{raw}/*.tar.gz")):
        mark = t + ".countix"
        if os.path.exists(mark):
            continue
        with tarfile.open(t) as tf:
            for m in tf:
                if m.isfile() and os.path.basename(m.name) in want:
                    tf.extract(m, ex, filter="data")
        open(mark, "w").close()
    srcs = sorted(glob.glob(f"{ex}/*/*.mp4"))
    out = f"{a.root}/clips/countix_train"
    jobs = [(s, f"{out}/{os.path.basename(os.path.dirname(s)).replace(' ', '_')}/{os.path.basename(s)}", None, None) for s in srcs]

    def meta(dst):
        b = os.path.basename(dst)[:-4]
        return dict(src="countix_train", label=os.path.basename(os.path.dirname(dst)).replace("_", " "), ytid=b[:11],
                    t0=int(b[12:18]), t1=int(b[19:25]))
    print(len(want), "wanted;", len(srcs), "found", flush=True)
    run_jobs(jobs, f"{out}/index.jsonl", a.workers, meta)


def cmd_charades(a):
    raw = f"{a.root}/raw/charades"
    if not os.path.exists(f"{raw}/Charades_v1_480/.done"):
        with zipfile.ZipFile(f"{raw}/Charades_v1_480.zip") as z:
            z.extractall(raw)
        open(f"{raw}/Charades_v1_480/.done", "w").close()
    if not os.path.exists(f"{raw}/Charades/.done"):
        with zipfile.ZipFile(f"{raw}/Charades.zip") as z:
            z.extractall(raw)
        open(f"{raw}/Charades/.done", "w").close()
    srcs = sorted(glob.glob(f"{raw}/Charades_v1_480/*.mp4"))
    # only what we use: Charades-STA test videos (eval) + train-split videos not on the exclusion list
    ann = f"{a.root}/raw/ann"
    sta_test = {l.split()[0] for l in open(f"{ann}/charades_sta_test.txt")}
    train = {l.split(",")[0] for l in open(f"{ann}/Charades/Charades_v1_train.csv")}
    ex = set(json.load(open(f"{a.root}/decon/exclusions.json"))["ids"].get("charades", []))
    vid = lambda s: os.path.basename(s)[:-4]
    srcs = [s for s in srcs if vid(s) in sta_test or (vid(s) in train and vid(s) not in ex)]
    print(len(srcs), "charades videos selected", flush=True)
    out = f"{a.root}/clips/charades"
    jobs = [(s, f"{out}/{os.path.basename(s)}", None, None) for s in srcs]
    run_jobs(jobs, f"{out}/index.jsonl", a.workers, lambda d: dict(src="charades", vid=os.path.basename(d)[:-4]))


def cmd_dir(a):
    srcs = sorted(p for ext in ("mp4", "MP4", "mkv", "webm", "avi", "mov") for p in glob.glob(f"{a.src}/**/*.{ext}", recursive=True))
    out = f"{a.root}/clips/{a.name}"
    jobs = []
    for s in srcs:
        rel = os.path.relpath(s, a.src).rsplit(".", 1)[0].replace("/", "__")
        if a.seg:
            d = probe(s)["dur"]
            for k in range(int(max(1, np.ceil(d / a.seg)))):
                jobs.append((s, f"{out}/{rel}__{k:04d}.mp4", k * a.seg, a.seg))
        else:
            jobs.append((s, f"{out}/{rel}.mp4", None, None))

    def meta(dst):
        b = os.path.basename(dst)[:-4]
        m = dict(src=a.name, vid=b.split("__")[-2] if a.seg else b)
        if a.seg:
            m["seg"] = int(b.split("__")[-1]); m["seg_t0"] = m["seg"] * a.seg
        return m
    run_jobs(jobs, f"{out}/index.jsonl", a.workers, meta)


def _hash_job(p):
    from vhash import frame_hashes
    try:
        return p, frame_hashes(p, 1.0)
    except Exception:
        return p, np.zeros(0, np.uint64)


def cmd_decon(a):
    from vhash import HashIndex
    z = np.load(a.hashes or f"{a.root}/decon/vi_hashes.npz")     # VI by default; --hashes for another benchmark
    idx = HashIndex(z["hash"], z["owner"])
    print("VI frames", len(z["hash"]), flush=True)
    names = a.name.split(",") if a.name else sorted(os.path.basename(os.path.dirname(f))
                                                     for f in glob.glob(f"{a.root}/clips/*/index.jsonl"))
    clips = [json.loads(l) for n in names for l in open(f"{a.root}/clips/{n}/index.jsonl")]
    tag = a.tag + "_".join(names)
    out = f"{a.root}/decon/clip_matches_{tag}.jsonl"
    done = {json.loads(l)["path"] for l in open(out)} if os.path.exists(out) else set()
    todo = [c["path"] for c in clips if c["path"] not in done]
    print(len(todo), "clips to hash", flush=True)
    with open(out, "a") as f, ProcessPoolExecutor(a.workers) as ex:
        for i, (p, h) in enumerate(ex.map(_hash_job, todo, chunksize=8)):
            m = idx.match(h)
            # a clip is contaminated when >= 2 of its 1-fps frames (or >= 30% for very short clips) match one VI video
            hit = {k: v for k, v in m.items() if v >= max(2, min(3, int(0.3 * len(h))))}
            f.write(json.dumps(dict(path=p, n=len(h), matches=hit)) + "\n")
            if i % 1000 == 0:
                f.flush(); print(i, flush=True)
    rows = [json.loads(l) for l in open(out)]
    bad = [r for r in rows if r["matches"]]
    base = os.path.dirname(a.root.rstrip("/"))
    for r in bad:
        r["rel"] = os.path.relpath(r["path"], base)       # "gv/clips/..." (host independent)
    json.dump(dict(names=names, against=os.path.basename(a.hashes or "vi_hashes.npz"), n_clips=len(rows), n_contaminated=len(bad), contaminated=bad),
              open(f"{a.root}/decon/clip_exclusions_{tag}.json", "w"), indent=1)
    print("contaminated", len(bad), "of", len(rows), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--root", default=os.path.expanduser("~/gv"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--src"); ap.add_argument("--name"); ap.add_argument("--seg", type=float, default=0)
    ap.add_argument("--hashes", default=""); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    dict(k700=cmd_k700, charades=cmd_charades, countix_train=cmd_countix_train, dir=cmd_dir, decon=cmd_decon)[a.cmd](a)
