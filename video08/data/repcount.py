"""RepCount-A (LLSP) from the lance mirror -> mp4 files + index with per-repetition end times (count-so-far labels).
python video08/data/repcount.py DATA_ROOT [split ...]   -> DATA_ROOT/raw/repcount_mp4/<split>/<id>.mp4, idx/repcount_<split>.jsonl"""
import json, os, sys
import lance

R = sys.argv[1]; splits = sys.argv[2:] or ["test", "validation", "train"]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from video08 import vio

ex = json.load(open(f"{R}/meta/decon/clip_exclusions_countix_train_repcount_train_repcount_validation_repcount_test.json"))
bad = {os.path.splitext(os.path.basename(c["rel"]))[0] for c in ex["contaminated"]}
for sp in splits:
    p = f"{R}/raw/repcount/data/{sp}.lance"
    if not os.path.exists(p):
        print("missing", sp); continue
    ds = lance.dataset(p); od = f"{R}/raw/repcount_mp4/{sp}"; os.makedirs(od, exist_ok=True)
    rows = []
    meta = ds.to_table(columns=["video_id", "action_type", "count", "cycle_bounds_json", "video_ext"]).to_pylist()
    for i, r in enumerate(meta):
            fn = f"{od}/{r['video_id']}.{r['video_ext']}"
            if not os.path.exists(fn):
                with ds.take_blobs("video_blob", indices=[i])[0] as bf:
                    open(fn, "wb").write(bf.read())
            try:
                fps, dur, w, h = vio.probe(fn)
            except Exception as e:
                print("bad video", fn, e); continue
            cb = json.loads(r["cycle_bounds_json"] or "[]")
            reps = [[cb[i] / fps, cb[i + 1] / fps] for i in range(0, len(cb) - 1, 2)]
            rows.append(dict(id=r["video_id"], split=sp, path=fn, action=r["action_type"], count=r["count"], fps=fps, dur=dur,
                             reps=reps, vi_contaminated=r["video_id"] in bad))
    os.makedirs(f"{R}/idx", exist_ok=True)
    with open(f"{R}/idx/repcount_{sp}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(sp, len(rows), "contaminated", sum(r["vi_contaminated"] for r in rows))
