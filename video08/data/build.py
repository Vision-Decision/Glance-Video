"""Build the indices and question sets for the 0.8B video experiments.   python video08/data/build.py DATA_ROOT

Outputs under DATA_ROOT/idx/:
  charades.jsonl     per video: id, split, path, length, actions [[cls, s, e]], sta [[sentence, s, e]], vi_excluded
  classes.json       Charades class id -> phrase
  pushup.jsonl       PushupBench (eval only): path, exercise, count
  splits.json        video-id sets: train (Charades train minus the Video-Index exclusion list), probe_val, eval (test)
Every eval video comes from the Charades *test* split (or PushupBench / RepCount test); training only ever sees Charades
train ids that are not in meta/decon/exclusions.json (ids used by Video-Index sources) -> dedup by video id.
"""
import csv, json, os, random, re, sys, glob

R = sys.argv[1]; A = f"{R}/meta/ann"; O = f"{R}/idx"; os.makedirs(O, exist_ok=True)
VID = f"{R}/raw/charades/Charades_v1_480"

cls = {}
for l in open(f"{A}/Charades/Charades_v1_classes.txt"):
    c, name = l.strip().split(" ", 1); cls[c] = name
json.dump(cls, open(f"{O}/classes.json", "w"), indent=0)

ex = json.load(open(f"{R}/meta/decon/exclusions.json"))
vi_ids = set(ex["ids"]["charades"])
bench = json.load(open(f"{R}/meta/decon/bench_charades.json"))
for k, v in bench.items():
    if not k.startswith("_"):
        vi_ids |= set(v)

sta = {}
for split in ("train", "test"):
    for l in open(f"{A}/charades_sta_{split}.txt"):
        head, sent = l.strip().split("##", 1); vid, s, e = head.split()
        sta.setdefault(vid, []).append([sent.strip(), float(s), float(e)])

rows = []
for split in ("train", "test"):
    for r in csv.DictReader(open(f"{A}/Charades/Charades_v1_{split}.csv")):
        acts = []
        for a in (r["actions"] or "").split(";"):
            if a.strip():
                c, s, e = a.split(); acts.append([c, float(s), float(e)])
        p = f"{VID}/{r['id']}.mp4"
        rows.append(dict(id=r["id"], split=split, path=p, exists=os.path.exists(p), length=float(r["length"]),
                         scene=r["scene"], actions=acts, sta=sta.get(r["id"], []), vi_excluded=r["id"] in vi_ids))
with open(f"{O}/charades.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

rng = random.Random(0)
train = sorted(r["id"] for r in rows if r["split"] == "train" and not r["vi_excluded"] and r["exists"] and r["actions"])
test = sorted(r["id"] for r in rows if r["split"] == "test" and r["exists"] and r["actions"])
rng.shuffle(train)
splits = dict(train=sorted(train[300:]), probe_val=sorted(train[:300]), eval=test)  # probe_val: hyper-parameters only
json.dump(splits, open(f"{O}/splits.json", "w"))
assert not (set(splits["train"]) | set(splits["probe_val"])) & set(splits["eval"])
assert not (set(splits["train"]) | set(splits["probe_val"])) & vi_ids

pu = []
for p in sorted(glob.glob(f"{R}/raw/pushup/*.mp4")):
    m = re.match(r"(.+?)_(\d+)_(.+)_\[(\d+)\]\.mp4$", os.path.basename(p))
    if m:
        pu.append(dict(path=p, id=f"{m.group(1)}_{m.group(2)}", exercise=m.group(3), count=int(m.group(4))))
with open(f"{O}/pushup.jsonl", "w") as f:
    for r in pu:
        f.write(json.dumps(r) + "\n")
print(json.dumps(dict(charades=len(rows), missing=sum(not r["exists"] for r in rows), vi_excluded_train=sum(
    r["vi_excluded"] for r in rows if r["split"] == "train"), train=len(splits["train"]), probe_val=len(splits["probe_val"]),
    eval=len(splits["eval"]), pushup=len(pu))))
