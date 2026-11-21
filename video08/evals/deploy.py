"""Deployment protocol streams: Charades videos concatenated into long streams, N rules per stream ("is someone <action>
right now?"), mostly negative, read every `--every` seconds. Saves per-stream score curves, labels for all 157 classes,
question-free pooled hidden states (for the linear probes) and timings.

python -m video08.evals.deploy --split eval --fps 2 --arm stream --streams 12 --minutes 10 --rules 16 --out DIR
  arm stream : streaming engine (bounded KV + GDN state)
  arm recent4: SimpleStream-style (fresh context with only the last 4 frame groups at every readout)
  arm none   : no questions, features only (probe training streams)
"""
import argparse, json, os, random, time
import numpy as np
import torch

from video08.stream import Stream, load, yes_no_ids, p_yes
from video08 import vio, feed

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="eval"); ap.add_argument("--fps", type=float, default=2); ap.add_argument("--arm", default="stream")
ap.add_argument("--streams", type=int, default=12); ap.add_argument("--minutes", type=float, default=10); ap.add_argument("--rules", type=int, default=16)
ap.add_argument("--every", type=float, default=0.5); ap.add_argument("--out", required=True); ap.add_argument("--size", type=int, default=256)
ap.add_argument("--data", default=os.path.expanduser("~/glance-video-exp/data")); ap.add_argument("--layers", default="6,12,18,24")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--chunk", type=int, default=0); ap.add_argument("--start", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
classes = json.load(open(f"{a.data}/idx/classes.json")); C = sorted(classes)
rows = {r["id"]: r for r in map(json.loads, open(f"{a.data}/idx/charades.jsonl"))}
pool = json.load(open(f"{a.data}/idx/splits.json"))[a.split]
layers = [int(x) for x in a.layers.split(",")] if a.layers else []
model, tok = load(); yid, nid = yes_no_ids(tok)
st = Stream(model, tok, readout_layers=layers)
chunk = a.chunk or max(1, int(round(a.fps * 0.5)))
used = set()


def phrase(c):
    p = classes[c]; return p[0].lower() + p[1:]


for s in range(a.start, a.start + a.streams):
    fn = f"{a.out}/stream{s:03d}_{a.arm}_fps{a.fps:g}.npz"
    rng = random.Random(a.seed * 7919 + s)  # same videos and rules for every arm / fps (paired)
    vids = []; tot = 0.0
    order = list(pool); rng.shuffle(order)
    for v in order:
        if tot >= a.minutes * 60:
            break
        vids.append(v); tot += rows[v]["length"]
    present = sorted({c for v in vids for c, _, _ in rows[v]["actions"]})
    k = min(len(present), a.rules // 2)
    rules = sorted(rng.sample(present, k) + rng.sample([c for c in C if c not in present], a.rules - k))
    if os.path.exists(fn):
        continue
    qs = [f"Is someone {phrase(c)} right now? Answer yes or no." for c in rules]
    st.reset(); t_off = 0.0; prev = None
    R = dict(t=[], p=[], ask_ms=[], feats={L: [] for L in layers}, seg=[])
    acc = {L: [] for L in layers}
    buf = []  # last 4 groups (recent4 arm)

    def cb(t):
        for L in layers:
            if acc[L]:
                R["feats"][L].append(torch.stack(acc[L]).mean(0).half().cpu().numpy()); acc[L].clear()
        R["t"].append(t)
        if a.arm == "none":
            return
        torch.cuda.synchronize(); t1 = time.perf_counter()
        if a.arm == "stream":
            ps = p_yes(st.ask(qs)["logits"], yid, nid)
        else:
            sub = Stream(model, tok); fr4 = torch.stack([b[0] for b in buf[-4:]]); tt = [b[1] for b in buf[-4:]]
            vis, grid = sub.encode(vio.pairs(fr4[1:], fr4[0]) if len(fr4) > 1 else vio.pairs(fr4))
            sub.write(vis, tt[-len(vis):], grid)
            ps = p_yes(sub.ask(qs)["logits"], yid, nid)
        torch.cuda.synchronize(); R["ask_ms"].append((time.perf_counter() - t1) * 1e3)
        R["p"].append(ps.float().cpu().numpy())

    for v in vids:
        r = rows[v]
        fr, tt = vio.decode(r["path"], a.fps, a.size)
        if len(fr) < 2:
            continue
        R["seg"].append([v, t_off, t_off + float(tt[-1]) + 1.0 / a.fps])
        if a.arm == "recent4":
            for x, t in zip(fr, tt):
                buf.append((x, t_off + float(t))); buf[:] = buf[-5:]
                # readout schedule identical to the stream arm
                if (round((t_off + float(t)) / a.every, 6) % 1) < 1e-6 and len(buf) >= 2:
                    cb(t_off + float(t))
            prev = fr[-1]
        else:
            def wcb(t):
                cb(t)
            # pooled hidden states of every group, accumulated between readouts
            T = len(tt); nxt = (np.floor(t_off / a.every) + 1) * a.every
            for i0 in range(0, T, 256):
                frs = fr[i0:i0 + 256]
                vis, grid = st.encode(vio.pairs(frs, prev if i0 == 0 else fr[i0 - 1]))
                for b in range(0, len(frs), chunk):
                    ts = [t_off + float(x) for x in tt[i0 + b:i0 + min(b + chunk, len(frs))]]
                    f = st.write(vis[b:b + chunk], ts, grid, return_feats=bool(layers))
                    for L in layers:
                        acc[L].extend(list(f[L]))
                    if ts[-1] + 1e-6 >= nxt:
                        cb(ts[-1]); nxt = (np.floor(ts[-1] / a.every + 1e-6) + 1) * a.every
            prev = fr[-1]
        t_off += float(tt[-1]) + 1.0 / a.fps
    # labels: per readout time, 157 classes "happening now"
    t = np.array(R["t"]); Y = np.zeros((len(t), len(C)), np.uint8)
    ci = {c: i for i, c in enumerate(C)}
    ivs = {c: [] for c in C}
    for v, a0, a1 in R["seg"]:
        for c, s0, e0 in rows[v]["actions"]:
            s1, e1 = a0 + s0, min(a0 + e0, a1)
            ivs[c].append([s1, e1]); Y[(t >= s1) & (t <= e1), ci[c]] = 1
    np.savez_compressed(fn, t=t, p=np.array(R["p"]) if R["p"] else np.zeros((0, len(rules))), rules=np.array(rules),
                        Y=Y, classes=np.array(C), seg=json.dumps(R["seg"]), ivs=json.dumps(ivs), ask_ms=np.array(R["ask_ms"]),
                        **{f"h{L}": np.array(R["feats"][L]) for L in layers})
    print("stream", s, "len", round(t_off), "s readouts", len(t), "rules", len(rules), flush=True)
