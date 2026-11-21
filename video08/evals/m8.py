"""M8: long-stream stability. One continuous stream of Charades test videos (default 60 min). Every `--probe_every` s a
word card is shown for 1 s and asked about 5 s later (shown word vs 3 unseen); every probe also asks 8 "right now"
action questions about the current clip. Logged over time: probe scores, per-write latency, KV length, position, RoPE
rebases, GPU memory.  --arm fresh replays the same probes on a fresh stream that starts `--warm` s before each probe
(paired control: a drift of the continuous arm relative to it is a long-stream effect).

python -m video08.evals.m8 --fps 30 --minutes 60 --arm long|fresh --out OUT.json [--chunk 1 for strict per-frame timing]
"""
import argparse, json, os, random, time
import numpy as np
import torch

from video08.stream import Stream, load, yes_no_ids, p_yes
from video08 import vio
from video08.events import card, WORDS, COLORS

ap = argparse.ArgumentParser()
ap.add_argument("--fps", type=float, default=30); ap.add_argument("--minutes", type=float, default=60); ap.add_argument("--arm", default="long")
ap.add_argument("--probe_every", type=float, default=120); ap.add_argument("--warm", type=float, default=60)
ap.add_argument("--out", required=True); ap.add_argument("--data", default=os.path.expanduser("~/glance-video-exp/data"))
ap.add_argument("--chunk", type=int, default=0); ap.add_argument("--size", type=int, default=256); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max_pos", type=int, default=120000)
a = ap.parse_args()
rows = {r["id"]: r for r in map(json.loads, open(f"{a.data}/idx/charades.jsonl"))}
classes = json.load(open(f"{a.data}/idx/classes.json")); C = sorted(classes)
ev = json.load(open(f"{a.data}/idx/splits.json"))["eval"]
rng = random.Random(a.seed); order = list(ev); rng.shuffle(order)
T_END = a.minutes * 60
# timeline: list of (video, t_off)
timeline, t = [], 0.0
for v in order:
    if t >= T_END:
        break
    timeline.append((v, t)); t += rows[v]["length"]
probes = []
for k, tp in enumerate(np.arange(a.probe_every / 2, T_END - 10, a.probe_every)):
    r2 = random.Random(a.seed * 1000 + k); w, *neg = r2.sample(WORDS, 4)
    probes.append(dict(k=k, t_card=float(tp), t_ask=float(tp) + 6.0, word=w, neg=neg, color=r2.choice(COLORS)))

model, tok = load(); yid, nid = yes_no_ids(tok)
st = Stream(model, tok, max_pos=a.max_pos)
chunk = a.chunk or max(1, int(round(a.fps * 0.5)))


def phrase(c):
    p = classes[c]; return p[0].lower() + p[1:]


def frames_between(t0, t1):
    """decoded frames of the timeline in [t0, t1) with card overlays, yielded per video piece"""
    for v, off in timeline:
        L = rows[v]["length"]
        if off + L <= t0 or off >= t1:
            continue
        a0, a1 = max(0.0, t0 - off), min(L, t1 - off)
        fr, tt = vio.decode(rows[v]["path"], a.fps, a.size, a0, a1)
        if len(fr) == 0:
            continue
        g = off + tt
        for p in probes:
            m = (g >= p["t_card"]) & (g < p["t_card"] + 1.0)
            if m.any():
                mm = torch.from_numpy(m); fr[mm] = card(fr[mm], p["word"], p["color"], a.size)
        yield v, off, fr, g


def ask_probe(p, v_now, t_now):
    qs = [f"Did the word '{w.upper()}' appear on screen at any point in the video? Answer yes or no." for w in [p["word"]] + p["neg"]]
    r = rows[v_now]; t_loc = t_now - dict(timeline)[v_now]
    now = sorted({c for c, s, e in r["actions"] if s <= t_loc <= e})
    r2 = random.Random(p["k"]); pos = now[:4]; neg = r2.sample([c for c in C if c not in {x[0] for x in r["actions"]}], 8 - len(pos))
    qa = [f"Is someone {phrase(c)} right now? Answer yes or no." for c in pos + neg]
    ps = p_yes(st.ask(qs + qa)["logits"], yid, nid).tolist()
    return dict(k=p["k"], t=t_now, word_p=ps[:4], act_p=ps[4:], act_y=[1] * len(pos) + [0] * len(neg))


log = dict(args=vars(a), probes=[], steps=[])


def run(t0, t1, pending):
    st.reset(); prev = None
    for v, off, fr, g in frames_between(t0, t1):
        for i0 in range(0, len(g), 256):
            frs = fr[i0:i0 + 256]
            torch.cuda.synchronize(); tv = time.perf_counter()
            vis, grid = st.encode(vio.pairs(frs, prev if i0 == 0 else fr[i0 - 1]))
            torch.cuda.synchronize(); vit_ms = (time.perf_counter() - tv) * 1e3 / len(frs)
            for b in range(0, len(frs), chunk):
                ts = [float(x) for x in g[i0 + b:i0 + min(b + chunk, len(frs))]]
                t1_ = time.perf_counter(); st.write(vis[b:b + chunk], ts, grid); torch.cuda.synchronize()
                dt = (time.perf_counter() - t1_) * 1e3
                if a.arm == "long":
                    log["steps"].append([ts[-1], dt, len(ts), vit_ms, st.kv_len(), st.pos, st.rebases,
                                         torch.cuda.memory_allocated() / 2**20])
                while pending and ts[-1] + 1e-6 >= pending[0]["t_ask"]:
                    p = pending.pop(0); torch.cuda.synchronize(); ta = time.perf_counter()
                    rec = ask_probe(p, v, ts[-1]); torch.cuda.synchronize()
                    rec["ask_ms"] = (time.perf_counter() - ta) * 1e3; rec.update(kv=st.kv_len(), pos=st.pos, rebases=st.rebases)
                    log["probes"].append(rec)
            prev = frs[-1]
        if not pending:
            break


t_start = time.perf_counter()
if a.arm == "long":
    run(0.0, T_END, list(probes))
else:
    for p in probes:
        run(max(0.0, p["t_card"] - a.warm), p["t_ask"] + 1.0, [p])
log["wall_s"] = time.perf_counter() - t_start
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(log, open(a.out, "w"))
print("done", a.arm, a.fps, "probes", len(log["probes"]), "wall", round(log["wall_s"]))
