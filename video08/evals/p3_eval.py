"""P3 evaluation: S1 heads vs the same frozen backbone's generative answers, on the SAME streamed state and the SAME
question token ids. Charades test (presence now / so far, temporal, which-one) and RepCount-A test (count so far).

python -m video08.evals.p3_eval --ckpt DIR/ckpt_final.pt --fps 2 --n 300 --out OUT.jsonl [--rep 152]
Per item: head score/prediction, generative score/prediction, plus timing of the N-question readout both ways.
"""
import argparse, json, os, random, time, zlib
import numpy as np
import torch

from video08.stream import Stream, load, yes_no_ids, p_yes
from video08 import vio, heads
from video08.tasks import Q
from video08.metrics import parse_span, parse_int

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--fps", type=float, default=2); ap.add_argument("--n", type=int, default=300)
ap.add_argument("--rep", type=int, default=152); ap.add_argument("--out", required=True); ap.add_argument("--gen", type=int, default=1)
ap.add_argument("--data", default=os.path.expanduser("~/glance-video-exp/data")); ap.add_argument("--size", type=int, default=256)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--sft", default="", help="generative-SFT checkpoint: also read SFT first-token scores and SFT decoding (same process)")
ap.add_argument("--split", default="eval", help="eval (Charades test) or probe_val (for temperature calibration)")
ap.add_argument("--sft_decode_frac", type=float, default=0.25, help="presence items also decoded by the SFT model (others: first token)")
a = ap.parse_args()
IDX = f"{a.data}/idx"
classes = json.load(open(f"{IDX}/classes.json"))
mapping = {l.split()[0]: l.split()[1:] for l in open(f"{a.data}/meta/ann/Charades/Charades_v1_mapping.txt")}
rows = {r["id"]: r for r in map(json.loads, open(f"{IDX}/charades.jsonl"))}
ev = json.load(open(f"{IDX}/splits.json"))[a.split]
R0 = random.Random(a.seed); ev = list(ev); R0.shuffle(ev); ev = ev[: a.n]
rep = [r for r in map(json.loads, open(f"{IDX}/repcount_test.jsonl"))][: a.rep] if a.rep and a.split == "eval" else []

model, tok = load(); yid, nid = yes_no_ids(tok)
ck = torch.load(a.ckpt, map_location="cuda", weights_only=False); A = ck["args"]
lora = heads.add_lora(model.model.language_model, r=A["lora"]) if A["lora"] else []
for p, q in zip(lora, ck["lora"]):
    p.data.copy_(q.cuda())
ro = heads.Readout(model.model.language_model.embed_tokens.weight, layers=[int(x) for x in A["layers"].split(",")],
                   slots=A["slots"], bins=A["bins"]).cuda().eval()
ro.load_state_dict(ck["ro"])
bank = {}
if a.sft:  # second adapter; heads.set_lora swaps the LoRA pointers between the S1b and the SFT adapter (free)
    assert lora, "the SFT comparison expects a LoRA readout checkpoint (S1b)"
    mods = heads.lora_modules(model.model.language_model)
    bank["head"] = [p for m in mods for p in (m.A, m.B)]
    sk = torch.load(a.sft, map_location="cuda", weights_only=False)
    bank["sft"] = [torch.nn.Parameter(q.cuda(), requires_grad=False) for q in sk["lora"]]
    assert len(bank["sft"]) == len(bank["head"])


def use(name):
    if bank:
        heads.set_lora(heads.lora_modules(model.model.language_model), bank[name])
st = Stream(model, tok); qb = Q(tok, classes, mapping)
letters = [tok.encode(x, add_special_tokens=False)[0] for x in "ABCD"]
none_id = tok.encode("none", add_special_tokens=False)[0]
done = {json.loads(l)["vid"] for l in open(a.out)} if os.path.exists(a.out) else set()
fo = open(a.out, "a")
jobs = [("charades", rows[v]) for v in ev] + [("count", r) for r in rep]


@torch.no_grad()
def gen_ids(ids, max_new=24):
    """greedy decode; uses the LoRA currently selected if heads.LORA['on'] (question block + answer tokens)"""
    """greedy decode from explicit ids (batch 1, forked)"""
    cache = st.fork(1); E = st.emb(torch.tensor([ids], device="cuda"))
    pos = torch.arange(st.pos, st.pos + len(ids), device="cuda").view(1, 1, -1).expand(3, 1, -1)
    o = st.lm(inputs_embeds=E, position_ids=pos, past_key_values=cache, use_cache=True)
    nxt = model.lm_head(o.last_hidden_state[:, -1]).argmax(-1); p = st.pos + len(ids); out = []
    for _ in range(max_new):
        if int(nxt) in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id):
            break
        out.append(int(nxt))
        o = st.lm(inputs_embeds=st.emb(nxt[:, None]), position_ids=torch.full((3, 1, 1), p, device="cuda"), past_key_values=cache, use_cache=True)
        nxt = model.lm_head(o.last_hidden_state[:, -1]).argmax(-1); p += 1
    return tok.decode(out, skip_special_tokens=True).strip()


@torch.no_grad()
def gen_logits(ids_list):
    N = len(ids_list); L = max(map(len, ids_list)); cache = st.fork(N)
    E = torch.zeros(N, L, 1024, device="cuda", dtype=st.dtype)
    for i, x in enumerate(ids_list):
        E[i, : len(x)] = st.emb(torch.tensor(x, device="cuda"))
    pos = torch.arange(st.pos, st.pos + L, device="cuda").view(1, 1, -1).expand(3, N, -1)
    o = st.lm(inputs_embeds=E, position_ids=pos, past_key_values=cache, use_cache=True)
    last = torch.tensor([len(x) - 1 for x in ids_list], device="cuda")
    return model.lm_head(o.last_hidden_state[torch.arange(N), last]).float()


for kind, r in jobs:
    if r["id"] in done:
        continue
    rng = random.Random(zlib.crc32(r["id"].encode()) + a.seed)
    srng = random.Random(zlib.crc32(r["id"].encode()) + 7777)  # SFT decode subsample: keeps the question sets identical to P3
    fr, tt = vio.decode(r["path"], a.fps, a.size, 0.0, 120.0)
    if len(fr) < 4:
        continue
    dur = float(tt[-1]); qt = sorted(rng.uniform(min(2.0, dur), dur) for _ in range(2)) + [dur]
    st.reset(); heads.LORA["on"] = False
    vis = torch.cat([st.encode(vio.pairs(fr[i:i + 256], fr[i - 1] if i else None))[0] for i in range(0, len(fr), 256)])
    grid = (a.size // 32, a.size // 32)
    chunk = max(1, int(round(a.fps * 0.5))); qi = 0; recs = []
    for b in range(0, len(tt), chunk):
        ts = [float(x) for x in tt[b:b + chunk]]
        with torch.no_grad():
            st.write(vis[b:b + chunk], ts, grid)
        while qi < len(qt) and ts[-1] + 1e-6 >= qt[qi]:
            t = ts[-1]; qi += 1
            items = qb.count(r, t, rng) if kind == "count" else qb.presence(r, t, rng) + qb.temporal(r, t, rng) + qb.choice(r, t, rng)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            with torch.no_grad():
                use("head"); heads.LORA["on"] = bool(lora)
                res = ro(st, [x["ids"] for x in items], [x.get("ends") for x in items])
                heads.LORA["on"] = False
            torch.cuda.synchronize(); head_ms = (time.perf_counter() - t1) * 1e3
            t2 = time.perf_counter()
            glog = gen_logits([x["ids"] for x in items])
            torch.cuda.synchronize(); genlogit_ms = (time.perf_counter() - t2) * 1e3
            slog = None
            if bank:
                use("sft"); heads.LORA["on"] = True; t3 = time.perf_counter()
                slog = gen_logits([x["ids"] for x in items]); torch.cuda.synchronize(); sftlogit_ms = (time.perf_counter() - t3) * 1e3
                heads.LORA["on"] = False
            for i, x in enumerate(items):
                o = dict(vid=r["id"], t=t, dur=dur, fps=a.fps, task=x["task"], head_ms=head_ms, genlogit_ms=genlogit_ms, nq=len(items))
                if slog is not None:
                    o.update(sftlogit_ms=sftlogit_ms)
                    heads.LORA["on"] = True
                    if x["task"] == "presence":
                        o["sft_first"] = float(p_yes(slog[i:i + 1], yid, nid))
                        if srng.random() < a.sft_decode_frac:
                            tg = time.perf_counter(); o["sft_text"] = gen_ids(x["ids"], 3); o["sft_ms"] = (time.perf_counter() - tg) * 1e3
                    elif x["task"] == "choice":
                        o["sft_first"] = slog[i, letters].tolist()
                        tg = time.perf_counter(); o["sft_text"] = gen_ids(x["ids"], 3); o["sft_ms"] = (time.perf_counter() - tg) * 1e3
                    elif x["task"] == "temporal":
                        o["sft_first_none"] = float(torch.softmax(slog[i], -1)[none_id])
                        tg = time.perf_counter(); o["sft_text"] = gen_ids(x["ids"], 24); o["sft_ms"] = (time.perf_counter() - tg) * 1e3
                        o["sft"] = parse_span(o["sft_text"])
                    elif x["task"] == "count":
                        tg = time.perf_counter(); o["sft_text"] = gen_ids(x["ids"], 4); o["sft_ms"] = (time.perf_counter() - tg) * 1e3
                        o["sft"] = parse_int(o["sft_text"])
                    heads.LORA["on"] = False
                if x["task"] == "presence":
                    o.update(kind=x["kind"], cls=x["cls"], y=x["y"], neg=x["neg"], head=float(torch.sigmoid(res["presence"][i])),
                             gen=float(p_yes(glog[i:i + 1], yid, nid)))
                elif x["task"] == "temporal":
                    sl = heads.decode_slots(res["obj"][i], res["start"][i], res["end"][i], ro.B)
                    o.update(cls=x["cls"], segs=[[s * t, e * t] for s, e in x["segs"]], head=[[p, s * t, e * t] for p, s, e in sl])
                    if a.gen and (x["cls"] == "sta" or not x["segs"] or rng.random() < 0.3):
                        tg = time.perf_counter(); ans = gen_ids(x["ids"]); o["gen_ms"] = (time.perf_counter() - tg) * 1e3
                        o.update(gen_text=ans, gen=parse_span(ans))
                elif x["task"] == "choice":
                    o.update(y=x["y"], head=res["choice"][i].float().tolist(), gen=glog[i, letters].tolist())
                elif x["task"] == "count":
                    o.update(y=x["y"], head=res["count"][i].float().softmax(-1).tolist())
                    if a.gen:
                        tg = time.perf_counter(); ans = gen_ids(x["ids"]); o["gen_ms"] = (time.perf_counter() - tg) * 1e3
                        o.update(gen_text=ans, gen=parse_int(ans))
                recs.append(o)
    for o in recs:
        fo.write(json.dumps(o) + "\n")
    fo.flush()
print("done", a)
