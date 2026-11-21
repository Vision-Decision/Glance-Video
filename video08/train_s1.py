"""P3 / S1: frozen Qwen3.5-0.8B, train only the readout (latent tokens + heads, optional question-block LoRA).
Training is in the inference shape: the clip is streamed (bounded KV + GDN state, LoRA off), question blocks are forked
at random query times (LoRA on), so the heads only ever see what the deployed stream would hold.

python -m video08.train_s1 --out DIR [--lora 16] [--steps 6000] [--fps 2:0.4,5:0.2,10:0.2,30:0.2]
Data: Charades train (minus Video-Index ids) -> presence / temporal / which-one; RepCount-A train (minus contaminated)
-> count. Checkpoints: DIR/ckpt_last.pt every --save s and at SIGTERM/SIGUSR1 (exit 3, resumable), DIR/ckpt_final.pt.
"""
import argparse, json, os, random, signal, sys, time
import numpy as np
import torch
import torch.nn.functional as F

from video08.stream import Stream, load
from video08 import vio, heads, feed
from video08.tasks import Q

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--data", default=os.path.expanduser("~/glance-video-exp/data"))
ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--lora", type=int, default=0)
ap.add_argument("--lora_lr", type=float, default=1e-4); ap.add_argument("--fps", default="2:0.4,5:0.2,10:0.2,30:0.2")
ap.add_argument("--layers", default="12,24"); ap.add_argument("--slots", type=int, default=4); ap.add_argument("--bins", type=int, default=32)
ap.add_argument("--p_count", type=float, default=0.2); ap.add_argument("--times", type=int, default=3)
ap.add_argument("--save", type=float, default=600); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--size", type=int, default=256)
ap.add_argument("--batch", type=int, default=8, help="videos per step, streamed in lockstep (same fps)")
ap.add_argument("--max_len", type=float, default=60.0)
ap.add_argument("--qbatch", type=int, default=8, help="question rows per forward/backward")
ap.add_argument("--write_s", type=float, default=0.5, help="seconds of frames per stream write (KV is evicted per write)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.manual_seed(a.seed); random.seed(a.seed)
fps_set = [(float(x.split(":")[0]), float(x.split(":")[1])) for x in a.fps.split(",")]

IDX = f"{a.data}/idx"
classes = json.load(open(f"{IDX}/classes.json"))
mapping = {l.split()[0]: l.split()[1:] for l in open(f"{a.data}/meta/ann/Charades/Charades_v1_mapping.txt")}
rows = {r["id"]: r for r in map(json.loads, open(f"{IDX}/charades.jsonl"))}
train_ids = json.load(open(f"{IDX}/splits.json"))["train"]
rep = [r for r in map(json.loads, open(f"{IDX}/repcount_train.jsonl"))] if os.path.exists(f"{IDX}/repcount_train.jsonl") else []
rep = [r for r in rep if not r["vi_contaminated"] and r["reps"]]
print("train videos: charades", len(train_ids), "repcount", len(rep), flush=True)


class DS(torch.utils.data.IterableDataset):
    def __iter__(self):
        wi = torch.utils.data.get_worker_info(); rng = random.Random(a.seed * 100003 + (wi.id if wi else 0) + int(time.time()))
        while True:
            fps = rng.choices([f for f, _ in fps_set], [w for _, w in fps_set])[0]
            vids = []
            while len(vids) < a.batch:
                if rep and rng.random() < a.p_count:
                    r = rng.choice(rep); kind = "count"
                else:
                    r = rows[rng.choice(train_ids)]; kind = "charades"
                try:
                    fr, tt = vio.decode(r["path"], fps, a.size, 0.0, a.max_len)
                except Exception as e:
                    print("decode fail", r["path"], e, flush=True); continue
                if len(fr) >= 4:
                    vids.append(dict(kind=kind, row=r, frames=fr, times=tt, seed=rng.randrange(1 << 30)))
            yield dict(fps=fps, vids=vids)


model, tok = load()
for p in model.parameters():
    p.requires_grad_(False)
lora_params = heads.add_lora(model.model.language_model, r=a.lora) if a.lora else []
ro = heads.Readout(model.model.language_model.embed_tokens.weight, layers=[int(x) for x in a.layers.split(",")],
                   slots=a.slots, bins=a.bins).cuda()
groups = [dict(params=list(ro.parameters()), lr=a.lr)] + ([dict(params=lora_params, lr=a.lora_lr)] if lora_params else [])
opt = torch.optim.AdamW(groups, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * max(0.05, 0.5 * (1 + np.cos(np.pi * min(1.0, s / a.steps)))))
st = Stream(model, tok)
qb = Q(tok, classes, mapping)
step = 0
ck = f"{a.out}/ckpt_last.pt"


def state():
    return dict(ro=ro.state_dict(), lora=[p.detach().cpu() for p in lora_params], opt=opt.state_dict(), sched=sched.state_dict(),
                step=step, args=vars(a))


if os.path.exists(f"{a.out}/ckpt_final.pt"):
    print("final exists"); sys.exit(0)
if os.path.exists(ck):
    s = torch.load(ck, map_location="cuda", weights_only=False)
    ro.load_state_dict(s["ro"]); [p.data.copy_(q) for p, q in zip(lora_params, s["lora"])]
    opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"]); step = s["step"]; print("resumed at", step, flush=True)
stop = {"flag": False}
for sg in (signal.SIGUSR1, signal.SIGTERM):
    signal.signal(sg, lambda *_: stop.__setitem__("flag", True))

def _winit(_):  # the queue stops a job with SIGTERM to its process group: workers ignore it, the main process saves
    signal.signal(signal.SIGTERM, signal.SIG_IGN); signal.signal(signal.SIGUSR1, signal.SIG_IGN)


dl = torch.utils.data.DataLoader(DS(), batch_size=None, num_workers=a.workers, prefetch_factor=2, persistent_workers=True,
                                 worker_init_fn=_winit)
logf = open(f"{a.out}/log.jsonl", "a")
t_save = time.time(); t0 = time.time(); agg = {}


def losses(res, items):
    L = {}
    pres = [i for i, x in enumerate(items) if x["task"] == "presence"]
    if pres:
        y = torch.tensor([items[i]["y"] for i in pres], device="cuda").float()
        w = torch.where(y > 0, 3.0, 1.0)  # positives are ~1/4 of the questions
        L["presence"] = (F.binary_cross_entropy_with_logits(res["presence"][pres], y, reduction="none") * w).mean()
    tem = [i for i, x in enumerate(items) if x["task"] == "temporal"]
    if tem:
        L["temporal"] = torch.stack([heads.slot_loss(res["obj"][i], res["start"][i], res["end"][i], items[i]["segs"], a.bins) for i in tem]).mean()
    cnt = [i for i, x in enumerate(items) if x["task"] == "count"]
    if cnt:
        y = torch.tensor([min(items[i]["y"], ro.C - 1) for i in cnt], device="cuda")
        lg = res["count"][cnt]
        tgt = torch.exp(-0.5 * ((torch.arange(ro.C, device="cuda")[None] - y[:, None]) / 0.7) ** 2); tgt = tgt / tgt.sum(-1, keepdim=True)
        L["count"] = -(F.log_softmax(lg, -1) * tgt).sum(-1).mean()
    ch = [i for i, x in enumerate(items) if x["task"] == "choice"]
    if ch:
        L["choice"] = torch.stack([F.cross_entropy(res["choice"][i][None], torch.tensor([items[i]["y"]], device="cuda")) for i in ch]).mean()
    return L


def batches():
    try:
        for b in dl:
            yield b
    except Exception as e:  # a worker died (e.g. killed with the group): save and leave as interrupted
        print("loader stopped:", repr(e)[:200], flush=True); stop["flag"] = True


t_prev = time.time()
for batch in batches():
    if step >= a.steps or stop["flag"]:
        break
    t_got = time.time(); tread = [0.0]
    fps = batch["fps"]; V = batch["vids"]; chunk = max(1, int(round(fps * a.write_s)))
    plans = []
    for v in V:
        rng = random.Random(v["seed"]); dur = float(v["times"][-1])
        qt = sorted(rng.uniform(min(2.0, dur), dur) for _ in range(a.times - 1)) + [dur]
        plans.append(dict(rng=rng, qt=qt, qi=0))
    heads.LORA["on"] = False; total = 0.0; parts = {}; nread = sum(len(p["qt"]) for p in plans)
    opt.zero_grad(set_to_none=True)

    def on_step(ts, active):
        global total
        t = ts[-1]
        for r, i in enumerate(active):
            P = plans[i]; v = V[i]
            while P["qi"] < len(P["qt"]) and t + 1e-6 >= P["qt"][P["qi"]]:
                P["qi"] += 1
                items = qb.count(v["row"], t, P["rng"]) if v["kind"] == "count" else \
                    qb.presence(v["row"], t, P["rng"]) + qb.temporal(v["row"], t, P["rng"]) + qb.choice(v["row"], t, P["rng"])
                if not items:
                    continue
                # questions in micro-batches: forward + backward each, so only one micro-batch of forked KV
                # (and SDPA's GQA-expanded copy kept for backward) is alive at a time
                mb = a.qbatch; tr0 = time.time()
                for q0 in range(0, len(items), mb):
                    sub = items[q0:q0 + mb]; w = len(sub) / len(items)
                    heads.LORA["on"] = bool(lora_params)
                    res = ro(st, [x["ids"] for x in sub], [x.get("ends") for x in sub], rows=[r] * len(sub))
                    heads.LORA["on"] = False
                    L = losses(res, sub)
                    if not L:
                        continue
                    loss = sum(L.values()) * w / nread
                    loss.backward(); total += float(loss)
                    for k, x in L.items():
                        parts[k] = parts.get(k, 0) + float(x) * w / nread
                tread[0] += time.time() - tr0

    feed.feed_batch(st, [(v["frames"], v["times"]) for v in V], chunk, lambda i, r: None, on_step=on_step)
    torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
    opt.step(); sched.step(); step += 1
    for k, v in parts.items():
        agg.setdefault(k, []).append(v)
    agg.setdefault("loss", []).append(total); agg.setdefault("fps", []).append(fps)
    t_end = time.time()
    agg.setdefault("s_wait", []).append(t_got - t_prev); agg.setdefault("s_read", []).append(tread[0])
    agg.setdefault("s_stream", []).append(t_end - t_got - tread[0]); t_prev = t_end
    if step % 25 == 0 or (step <= 50 and step % 5 == 0):
        rec = dict(step=step, t=round(time.time() - t0), lr=sched.get_last_lr()[0], **{k: float(np.mean(v)) for k, v in agg.items()})
        logf.write(json.dumps(rec) + "\n"); logf.flush(); print(rec, flush=True); agg = {}
    if time.time() - t_save > a.save:
        torch.save(state(), ck + ".tmp"); os.replace(ck + ".tmp", ck); t_save = time.time()

torch.save(state(), ck + ".tmp"); os.replace(ck + ".tmp", ck)
if step >= a.steps:
    torch.save(dict(ro=ro.state_dict(), lora=[p.detach().cpu() for p in lora_params], step=step, args=vars(a)), f"{a.out}/ckpt_final.pt")
    print("final", step, flush=True); os._exit(0)
print("stopped at", step, flush=True); os._exit(3)  # skip DataLoader teardown (it can hang with workers ignoring TERM)
