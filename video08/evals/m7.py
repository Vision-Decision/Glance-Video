"""M7: GDN memory span. A short synthetic event (a word card over the frame for `--ev_s` seconds) is inserted into a
background stream of Charades test videos at t0; after delta seconds we ask "did the word X appear?" for the shown word
and 3 unseen words, at every delta of the grid, from ONE pass over the stream (readouts are forked, never written back).

python -m video08.evals.m7 --fps 30 --kv layered|short|full --trials 60 --out OUT.jsonl
  layered: 4 s full rate + 4-60 s at 2 fps in KV (the deployment window); short: KV holds 4 s only, so for delta > 4 s the
  event can reach the answer only through the GDN state; full: unbounded KV (reference, fps 2 only).
"""
import argparse, json, os, random, time
import numpy as np
import torch

from video08.stream import Stream, load, yes_no_ids, p_yes
from video08 import vio, feed
from video08.events import card, WORDS, COLORS

ap = argparse.ArgumentParser()
ap.add_argument("--fps", type=float, default=2); ap.add_argument("--kv", default="layered"); ap.add_argument("--trials", type=int, default=60)
ap.add_argument("--out", required=True); ap.add_argument("--data", default=os.path.expanduser("~/glance-video-exp/data"))
ap.add_argument("--t0", type=float, default=20.0); ap.add_argument("--ev_s", type=float, default=1.0)
ap.add_argument("--deltas", default="0,1,2,4,8,16,32,64,128,256"); ap.add_argument("--size", type=int, default=256)
ap.add_argument("--chunk", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
deltas = [float(x) for x in a.deltas.split(",")]
T_END = a.t0 + a.ev_s + max(deltas) + 0.5

rows = {r["id"]: r for r in map(json.loads, open(f"{a.data}/idx/charades.jsonl"))}
ev_ids = json.load(open(f"{a.data}/idx/splits.json"))["eval"]
model, tok = load(); yid, nid = yes_no_ids(tok)
kw = dict(layered=dict(full_s=4, sub_s=60), short=dict(full_s=4, sub_s=4), full=dict(keep_all=True))[a.kv]
st = Stream(model, tok, **kw)
chunk = a.chunk or max(1, int(round(a.fps * 0.5)))
done = {json.loads(l)["trial"] for l in open(a.out)} if os.path.exists(a.out) else set()
fo = open(a.out, "a")

for trial in range(a.trials):
    rng = random.Random(1000 * a.seed + trial)  # same videos / words / colours for every fps and kv arm (paired)
    vids = rng.sample(ev_ids, 40)
    word, *neg = rng.sample(WORDS, 4); color = rng.choice(COLORS)
    if trial in done:
        continue
    qs = [f"Did the word '{w.upper()}' appear on screen at any point in the video? Answer yes or no." for w in [word] + neg]
    st.reset(); t_off = 0.0; prev = None; res = []; t_start = time.perf_counter()
    targets = sorted(a.t0 + a.ev_s + d for d in deltas)  # readout times (stream seconds)
    ti = 0

    def cb(t):
        global ti
        while ti < len(targets) and t + 1e-6 >= targets[ti]:
            ps = p_yes(st.ask(qs)["logits"], yid, nid).tolist()
            res.append(dict(delta=round(targets[ti] - a.t0 - a.ev_s, 3), t=t, p=ps, kv=st.kv_len())); ti += 1

    for v in vids:
        if t_off >= T_END:
            break
        r = rows[v]
        fr, tt = vio.decode(r["path"], a.fps, a.size, 0.0, min(r["length"], T_END - t_off))
        if len(fr) < 2:
            continue
        g = t_off + tt
        m = (g >= a.t0) & (g < a.t0 + a.ev_s)
        if m.any():
            fr[torch.from_numpy(m)] = card(fr[torch.from_numpy(m)], word, color, a.size)
        prev, _, _ = feed.feed(st, fr, tt, chunk=chunk, t_offset=t_off, prev=prev, cb=cb)
        t_off = float(g[-1]) + 1.0 / a.fps
    fo.write(json.dumps(dict(trial=trial, fps=a.fps, kv=a.kv, word=word, neg=neg, res=res,
                             wall_s=time.perf_counter() - t_start, rebases=st.rebases)) + "\n"); fo.flush()
print("done", a)
