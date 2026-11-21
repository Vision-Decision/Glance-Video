"""N-question throughput on a warm 30 fps stream (60 s of Charades test, deployment KV window).
  write   : one frame group (strict per-frame) latency
  yesno   : N question blocks in ONE forked forward, answer = yes/no logit (generative, best case)
  gen     : N questions answered one by one with greedy decoding (generative, as deployed by most systems)
  head    : N question blocks + latent tokens + heads (S1), one forward  (--ckpt)
python -m video08.evals.throughput --out OUT.json [--ckpt CKPT]
"""
import argparse, json, os, time
import numpy as np
import torch
from video08.stream import Stream, load, yes_no_ids, p_yes
from video08 import vio, feed, heads

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--ckpt", default=""); ap.add_argument("--fps", type=float, default=30)
ap.add_argument("--data", default=os.path.expanduser("~/glance-video-exp/data")); ap.add_argument("--reps", type=int, default=20)
a = ap.parse_args()
rows = {r["id"]: r for r in map(json.loads, open(f"{a.data}/idx/charades.jsonl"))}
ev = json.load(open(f"{a.data}/idx/splits.json"))["eval"]
model, tok = load(); yid, nid = yes_no_ids(tok)
ro = None
if a.ckpt:
    ck = torch.load(a.ckpt, map_location="cuda", weights_only=False); A = ck["args"]
    lora = heads.add_lora(model.model.language_model, r=A["lora"]) if A["lora"] else []
    for p, q in zip(lora, ck["lora"]):
        p.data.copy_(q.cuda())
    ro = heads.Readout(model.model.language_model.embed_tokens.weight, layers=[int(x) for x in A["layers"].split(",")],
                       slots=A["slots"], bins=A["bins"]).cuda().eval(); ro.load_state_dict(ck["ro"])
st = Stream(model, tok)
t_off, prev = 0.0, None
for v in ev[:4]:
    fr, tt = vio.decode(rows[v]["path"], a.fps, 256)
    prev, _, _ = feed.feed(st, fr, tt, chunk=15, t_offset=t_off, prev=prev); t_off += float(tt[-1]) + 1 / a.fps
    if t_off > 60:
        break
classes = json.load(open(f"{a.data}/idx/classes.json")); C = sorted(classes)
qs = [f"Is someone {classes[c][0].lower() + classes[c][1:]} right now? Answer yes or no." for c in C]
tq = [f"When has someone been {classes[c][0].lower() + classes[c][1:]} so far? Give the start and end times in seconds." for c in C]


def timeit(fn, reps):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts)), float(np.percentile(ts, 90))


out = dict(kv=st.kv_len(), t=t_off)
vis, grid = st.encode(vio.pairs(fr[-2:], fr[-3]))
def one_write():
    st.write(vis[:1], [st.now + 1 / a.fps], grid)
out["write_ms"] = timeit(one_write, 60)
for N in (1, 2, 4, 8, 16, 32, 64):
    r = dict(yesno=timeit(lambda: p_yes(st.ask(qs[:N])["logits"], yid, nid), a.reps))
    if N <= 16:
        r["gen"] = timeit(lambda: [st.generate(q, max_new=16) for q in tq[:N]], max(3, a.reps // 5))
    if ro is not None:
        ids = [st.question_ids(q) for q in qs[:N]]
        def hd():
            heads.LORA["on"] = bool(A["lora"]); ro(st, ids); heads.LORA["on"] = False
        with torch.no_grad():
            r["head"] = timeit(hd, a.reps)
    out[N] = r; print(N, r, flush=True)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
