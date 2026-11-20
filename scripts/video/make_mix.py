"""Training mixture for Glance-Video: video records + v1.1 replay (text + COCO box) to avoid forgetting.

  python scripts/video/make_mix.py OUT  FILE:N [FILE:N ...]     (N = records to take; 0 = all; sampled without replacement,
                                                                 with replacement if N > len)
  e.g. make_mix.py /content/data/gv/records/train_mix.jsonl  gv/records/train_k700.jsonl:0 .../train_charades.jsonl:0 \
       .../train_countix.jsonl:0 .../train_repcount.jsonl:0 /content/data/gv/replay/v1_text.jsonl:20000 .../v1_coco.jsonl:6000
"""
import json, random, sys
from collections import Counter

out, specs = sys.argv[1], sys.argv[2:]
rng = random.Random(0)
recs, stats = [], Counter()
for sp in specs:
    f, n = sp.rsplit(":", 1)
    n = int(n)
    ls = open(f).readlines()
    if n == 0 or n == len(ls):
        take = ls
    elif n < len(ls):
        take = rng.sample(ls, n)
    else:
        take = ls * (n // len(ls)) + rng.sample(ls, n % len(ls))
    recs += take
    stats[f.rsplit("/", 1)[-1]] = len(take)
rng.shuffle(recs)
with open(out, "w") as fo:
    fo.writelines(recs)
nq = sum(len(json.loads(l)["questions"]) for l in recs)
print(out, len(recs), "records", nq, "questions", dict(stats))
