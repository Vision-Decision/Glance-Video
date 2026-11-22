"""Shrink ~/glance-video-exp on the shared A100 host to <= 40 GB (2026-11-22, after the disk filled up).
1. verify that every source used by training / evaluation has its 256x256 cache (exists, non-empty, decodable)
2. trim the fps-2 uint8 frame cache to the evaluation videos actually used (M6/P3 first 300, deployment streams)
3. delete the raw sources that are cached (Charades, RepCount lance + mp4, PushupBench) and all B-group raw data
   (HC-STVG, MeViS, LaSOT: B runs on the G4)
python video08/data/cleanup_a100.py DATA [--dry]"""
import glob, json, os, random, shutil, sys
R = sys.argv[1]; DRY = "--dry" in sys.argv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import av


def du(p):
    t = 0
    for root, _, fs in os.walk(p):
        for f in fs:
            try:
                t += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return t / 2**30


def cached_ok(src):
    c = src.replace("/raw/", "/cache256/", 1)
    if not os.path.exists(c) or os.path.getsize(c) < 1000:
        return False
    try:
        with av.open(c) as f:
            return f.streams.video[0].frames > 0 or f.duration is not None
    except Exception:
        return False


before = du(f"{R}/..")
idx = f"{R}/idx"
rows = [json.loads(l) for l in open(f"{idx}/charades.jsonl")]
sp = json.load(open(f"{idx}/splits.json"))
need = {r["id"]: r["path"] for r in rows}
srcs = [need[i] for k in ("train", "probe_val", "eval") for i in sp[k]]
for s in ("train", "validation", "test"):
    f = f"{idx}/repcount_{s}.jsonl"
    if os.path.exists(f):
        srcs += [json.loads(l)["path"] for l in open(f)]
srcs += [json.loads(l)["path"] for l in open(f"{idx}/pushup.jsonl")]
missing = [s for s in srcs if not cached_ok(s)]
print("sources", len(srcs), "missing/broken cache", len(missing), missing[:5], flush=True)
if missing:
    if DRY:
        print("(dry) would stop here")
    else:
        raise SystemExit("cache incomplete: not deleting raw data")
# fps-2 npy: keep the evaluation videos actually used
ev = list(sp["eval"]); random.Random(0).shuffle(ev); keep = set(ev[:300])
for s in range(12):  # deployment streams (evals/deploy.py: rng seed 0*7919+s, 10-min streams)
    rng = random.Random(s); order = list(sp["eval"]); rng.shuffle(order); tot = 0
    for v in order:
        if tot >= 600:
            break
        keep.add(v); tot += next(r["length"] for r in rows if r["id"] == v)
keep_paths = {need[i] for i in keep} | {json.loads(l)["path"] for l in open(f"{idx}/pushup.jsonl")}
keep_paths |= {json.loads(l)["path"] for l in open(f"{idx}/repcount_test.jsonl")}
n_rm = 0
for f in glob.glob(f"{R}/cache_fps2/**/*.npy", recursive=True):
    if f.endswith(".t.npy"):
        continue
    src = f.replace("/cache_fps2/", "/raw/", 1)[:-4]
    if not any(src + ext in keep_paths for ext in (".mp4", ".avi", ".mkv", ".webm", ".mov")):
        n_rm += 1
        if not DRY:
            os.remove(f); os.remove(f[:-4] + ".t.npy")
print("fps2 npy removed", n_rm, "kept for", len(keep_paths), "eval videos", flush=True)
for d in ("charades", "charades_ai2", "charades_zip", "repcount", "repcount_mp4", "pushup", "hcstvg2", "mevis", "lasot", "refcoco_imgs"):
    p = f"{R}/raw/{d}"
    if os.path.exists(p):
        print("rm", p, round(du(p), 1), "GB", flush=True)
        if not DRY:
            shutil.rmtree(p)
after = du(f"{R}/..")
print(f"glance-video-exp: {before:.1f} GB -> {after:.1f} GB", flush=True)
