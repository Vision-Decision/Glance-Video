"""Glance-Video offline evaluation (Colab GPU).

  python scripts/video/eval_video.py typed --recs A.jsonl,B.jsonl [--calib C.jsonl]   # held-out generator tasks
  python scripts/video/eval_video.py sta   [--n 600]                                    # Charades-STA test (scan)
  python scripts/video/eval_video.py repcount                                           # RepCount-A test
  python scripts/video/eval_video.py vi --which v1_1|v1_0 [--n 0]                       # Video-Index choose-only
common: --ckpt PATH (Glance-Video / v1 checkpoint) | --base (zero-shot Qwen3.5-4B: LM readout, causal mask)
        --data_root /content  --out results.json  --rows rows.jsonl.gz

Model input is always a native video prefix (dv/data/video.py). Frames for STA / RepCount / VI come from ffmpeg and are
fed through an in-process clip cache (no files written). Video-Index videos are evaluation-only: they are fetched on the
VM from the official HF datasets and never stored anywhere else.
"""
import argparse, gzip, json, math, os, random, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch

from dv.config import Config
from dv.data import video as V
from dv.data.mixed import MixedDataset, collate_mixed, TokenBudgetSampler, LETTERS
from dv.model.build import build, load_ckpt
from dv.model.v1forward import run_model_v1
from train import to_dev

LETTER = {c: i for i, c in enumerate(LETTERS)}


# ------------------------------------------------------------------------------------------------ model
def load(a):
    if a.base:
        cfg = Config().update(dict(task="v1", attn_mask="causal", lora_r=0))
        sd = None
    else:
        sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        cfg = Config().update(sd["cfg"])
    cfg = cfg.update(dict(s.split("=", 1) for s in a.set), from_str=True)
    cfg.data_root = a.data_root
    net, heads, proc = build(cfg, "cuda", lora=not a.base)
    if not a.base:
        load_ckpt(a.ckpt, net, heads)
    net.eval(); heads.eval()
    temps = (sd or {}).get("extra", {}).get("temps", {}) if sd else {}
    return net, heads, proc, cfg, temps


@torch.no_grad()
def predict(M, recs, workers=6, prefix_budget=24576, q_budget=256):
    """-> list of rows (one per question): dict(rec_i, q_i, type, logits, target, task)"""
    net, heads, proc, cfg, _ = M
    for i, r in enumerate(recs):
        r["_i"] = i
        for q in r["questions"]:          # eval-only questions without a target (VI, STA scan): dummy target
            if "target" not in q:
                q["_notarget"] = True
                q["target"] = 0.5 if q["type"] == "noul" else [1.0 / len(q["options"])] * len(q["options"])
    ds = MixedDataset(None, cfg.data_root, proc, cfg, train=False, recs=recs)
    tok = ds.tok
    sampler = TokenBudgetSampler(ds, prefix_budget, q_budget, 64, shuffle=False, pool=len(ds) + 1)
    dl = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=workers,
                                     collate_fn=lambda it: collate_mixed(it, tok))
    rows = []
    for batch in dl:
        meta = batch.pop("meta")
        b = to_dev(batch, "cuda")
        res = run_model_v1(net, heads, b, train=False)
        tl = res["typed_logits"].float().cpu()
        k = 0
        for mt in meta:
            for j, q in enumerate(mt["qs"]):
                lg = [float(x) for x in tl[k] if x > -1e30]
                tgt = None if q.get("_notarget") else q.get("target")
                if q["type"] == "noul" and tgt is not None:
                    tgt = [1 - tgt, tgt]
                rows.append(dict(rec_i=mt["rec"]["_i"], q_i=j, type=q["type"], task=q.get("task", q["type"]), logits=lg,
                                 target=tgt, **{kk: q[kk] for kk in ("key", "count", "cand") if kk in q}))
                k += 1
    return rows


def probs(lg, T=1.0):
    x = np.asarray(lg) / T
    e = np.exp(x - x.max())
    return e / e.sum()


def ece(conf, corr, n=15):
    conf, corr = np.asarray(conf), np.asarray(corr)
    e, edges = 0.0, np.linspace(0, 1, n + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - corr[m].mean())
    return float(e)


def auroc(p, y):
    p, y = np.asarray(p), np.asarray(y)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return None
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fit_T(rows):
    from eval import fit_temperature
    return fit_temperature([r["logits"] for r in rows], [r["target"] for r in rows])[0]


def summarize(rows, temps=None):
    out = {}
    for task in sorted({r["task"] for r in rows}):
        rr = [r for r in rows if r["task"] == task and r["target"] is not None]
        if not rr:
            continue
        T = (temps or {}).get(rr[0]["type"], 1.0)
        P = [probs(r["logits"], T) for r in rr]
        y = [int(np.argmax(r["target"])) for r in rr]
        corr = [float(p.argmax() == t) for p, t in zip(P, y)]
        s = dict(n=len(rr), acc=float(np.mean(corr)), ece=ece([p.max() for p in P], corr),
                 brier=float(np.mean([((p - np.asarray(r["target"])) ** 2).sum() for p, r in zip(P, rr)])))
        if rr[0]["type"] == "noul":
            s["auroc"] = auroc([p[1] for p in P], y)
            s["pos_rate"] = float(np.mean(y))
        if rr[0]["type"] == "choice":
            s["chance"] = float(np.mean([1 / len(p) for p in P]))
        out[task] = s
    # pooled yes/no groups (positives and their negatives live in different tasks)
    for g, ts in (("event_all", ("event", "event_neg", "event_objneg", "event_rev_neg", "event_tneg")),
                  ("event_hardneg", ("event", "event_rev_neg", "event_tneg"))):
        rr = [r for r in rows if r["task"] in ts and r["target"] is not None]
        if rr:
            P = [probs(r["logits"], (temps or {}).get("noul", 1.0))[1] for r in rr]
            y = [int(np.argmax(r["target"])) for r in rr]
            out[g] = dict(n=len(rr), auroc=auroc(P, y), acc=float(np.mean([(p > 0.5) == t for p, t in zip(P, y)])))
    return out


# ------------------------------------------------------------------------------------------------ frames
def probe(path):
    o = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height:format=duration", "-of", "json", path],
                                  capture_output=True, text=True).stdout)
    s = o["streams"][0]
    return float(o["format"]["duration"]), int(s["width"]), int(s["height"])


def frames_at(path, t0, n, fps, short=256):
    """n frames from t0 at fps (seek first; fast for long videos) -> uint8 [n,H,W,3] (may return fewer at the end)"""
    dur, W, H = probe(path)
    w, h = (short, int(round(H * short / W / 2)) * 2) if W <= H else (int(round(W * short / H / 2)) * 2, short)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{max(0.0, t0):.3f}", "-i", path, "-t",
                          f"{n / fps + 0.5:.3f}", "-an", "-vf", f"fps={fps},scale={w}:{h}", "-frames:v", str(n),
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    k = len(raw) // (w * h * 3)
    return np.frombuffer(raw[: k * w * h * 3], np.uint8).reshape(k, h, w, 3)


def frames_uniform(path, n, short=256):
    """n frames uniformly over the whole video (one seek per frame)"""
    dur, W, H = probe(path)
    ts = [(i + 0.5) * dur / n for i in range(n)]
    fr = [frames_at(path, t, 1, 1.0, short) for t in ts]
    fr = [f[0] for f in fr if len(f)]
    if len(fr) < 2:   # seeking failed (odd container / timestamps): decode sequentially at n/dur fps instead
        fr = list(frames_at(path, 0.0, n, max(n / max(dur, 1e-3), 0.01), short))
    if not fr:
        raise ValueError("no frames decoded")
    return np.stack(fr), dur / n


def clip_rec(key, frames, dt, qs, max_vis=96, **kw):
    V._CACHE[f"/mem/{key}"] = frames
    return dict(video=key, frames=list(range(len(frames))), dt=dt, max_vis=max_vis,
                plen=V.est_prefix_tokens(len(frames), frames.shape[2], frames.shape[1], max_vis), questions=qs, **kw)


def run_mem(M, recs, workers=0):
    """records whose frames sit in V._CACHE['/mem/<key>'] (single-process loader: the cache is in this process)"""
    cfg = M[3]
    old = cfg.data_root
    cfg.data_root = "/mem"
    V._CACHE_MAX = 10 ** 9
    try:
        return predict(M, recs, workers=0)
    finally:
        cfg.data_root = old


# ------------------------------------------------------------------------------------------------ tasks
def cmd_typed(a, M):
    recs = [json.loads(l) for f in a.recs.split(",") for l in open(f)]
    if a.n:
        random.Random(0).shuffle(recs); recs = recs[: a.n]
    t0 = time.time()
    rows = predict(M, recs)
    res = dict(raw=summarize(rows), sec=time.time() - t0, n_recs=len(recs))
    if a.calib:
        crecs = [json.loads(l) for l in open(a.calib)][:3000]
        crows = predict(M, crecs)
        temps = {t: fit_T([r for r in crows if r["type"] == t and r["target"] is not None])
                 for t in ("noul", "choice", "score") if any(r["type"] == t for r in crows)}
        res["temps"] = temps
        res["cal"] = summarize(rows, temps)
    return res, rows


def maxsum_interval(p, tau):
    best, cur, s, bs, be = -1e9, 0.0, 0, 0, 0
    for i, x in enumerate(p):
        if cur <= 0:
            cur, s = 0.0, i
        cur += x - tau
        if cur > best:
            best, bs, be = cur, s, i + 1
    return bs, be


def iou(a, b):
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    u = max(a[1], b[1]) - min(a[0], b[0])
    return i / u if u > 0 else 0.0


def cmd_sta(a, M):
    """Charades-STA test: 4 s clips at 2 fps every 1 s; one prefix per clip, all sentences of the video as blocks;
    per-second P(t) = mean over covering clips; interval = max-sum segment of P - tau (tau = 0.5 abs, and relative)."""
    ann = [l.strip().split("##") for l in open(f"{a.gv_root}/raw/ann/charades_sta_test.txt")]
    byv = {}
    for head, s in ann:
        v, t0, t1 = head.split()
        byv.setdefault(v, []).append((float(t0), float(t1), s.strip().rstrip(".")))
    vids = sorted(byv)
    random.Random(0).shuffle(vids)
    idx = {json.loads(l)["vid"]: json.loads(l) for l in open(f"{a.data_root}/gv/clips/charades/index.jsonl")}
    out, nq = [], 0
    for v in vids:
        if nq >= a.n:
            break
        c = idx.get(v)
        if c is None:
            continue
        path = f"{a.data_root}/gv/clips/charades/{v}.mp4"
        arr = V.decode_clip(path)
        dur = len(arr) / 8.0
        sents = byv[v]
        recs = []
        starts = np.arange(0, max(1e-3, dur - 4.0) + 1e-6, 1.0)
        for t0 in starts:
            fr = arr[int(t0 * 8): int((t0 + 4.0) * 8): 4]
            if len(fr) < 4:
                continue
            qs = [dict(type="noul", instr=f"Does this happen in this clip: {s}?", task="sta") for _, _, s in sents]
            recs.append(clip_rec(f"{v}_{t0:.0f}", fr, 0.5, qs, t0=float(t0)))
        rows = run_mem(M, recs)
        T = M[4].get("noul", 1.0)
        P = np.zeros((len(sents), int(math.ceil(dur)) + 1)); C = np.zeros_like(P)
        pclip = [[] for _ in sents]       # clip-level P(yes) per query: [clip start s, p] (for scan-engine validation)
        for r in rows:
            t0 = recs[r["rec_i"]]["t0"]
            pp = probs(r["logits"], T)[1]
            pclip[r["q_i"]].append([t0, round(float(pp), 4)])
            P[r["q_i"], int(t0): int(t0) + 4] += pp; C[r["q_i"], int(t0): int(t0) + 4] += 1
        P = P / np.maximum(C, 1)
        for qi, (g0, g1, s) in enumerate(sents):
            p = P[qi, : int(math.ceil(dur))]
            res = dict(vid=v, s=s, gt=[g0, g1], dur=dur, clip_s=4.0, clip_fps=2.0, stride_s=1.0,
                       p_clip=sorted(pclip[qi]), p_sec=[round(float(x), 4) for x in p])
            for name, tau in (("abs", 0.5), ("rel", 0.5 * (p.max() + p.mean()))):
                bs, be = maxsum_interval(p, tau)
                pr = [float(bs), float(min(be, dur))]
                res[name] = dict(pred=pr, iou=iou(pr, [g0, g1]))
            res["pmax"] = float(p.max())
            out.append(res); nq += 1
        for r in recs:
            V._CACHE.pop(f"/mem/{r['video']}", None)
    s = {}
    for name in ("abs", "rel"):
        ious = np.array([o[name]["iou"] for o in out])
        s[name] = {"R1@0.3": float((ious >= 0.3).mean()), "R1@0.5": float((ious >= 0.5).mean()),
                   "R1@0.7": float((ious >= 0.7).mean()), "mIoU": float(ious.mean())}
    return dict(n=len(out), **s), out


def cmd_repcount(a, M):
    """RepCount-A test (prepared by prep_repcount.py: gv/clips/repcount_test/*.mp4 + index with count, action).
    direct: one prefix, <= 48 frames uniformly at >= 2.5 frames per expected period is impossible to know, so we use
    window counting (as trained): non-overlapping 4 s windows at 8 fps (32 frames), "how many repetitions are completed
    in this clip", choice over 0..12 -> expected count (and argmax), summed over windows. MAE = mean |pred-gt|/gt, OBO."""
    idx = [json.loads(l) for l in open(f"{a.data_root}/gv/clips/repcount_test/index.jsonl")]
    out = []
    opts = [str(i) for i in range(13)]
    for c in idx[: a.n or None]:
        arr = V.decode_clip(f"{a.data_root}/{c['rel']}")
        n = len(arr)
        recs = []
        for k, s in enumerate(range(0, n, 32)):
            fr = arr[s: s + 32]
            if len(fr) < 8:
                continue
            act = c["action"].replace("_", " ")
            recs.append(clip_rec(f"rc{c['id']}_{k}", fr, 0.125, [
                dict(type="choice", instr=f"How many repetitions of this repeated action ({act}) are completed in this clip?",
                     options=opts, task="rep_window")], max_vis=48))
        rows = run_mem(M, recs)
        T = M[4].get("choice", 1.0)
        ev = sum(float(np.dot(probs(r["logits"], T), np.arange(13))) for r in rows)
        am = sum(int(np.argmax(r["logits"])) for r in rows)
        out.append(dict(id=c["id"], gt=c["count"], pred_ev=ev, pred_argmax=am))
        for r in recs:
            V._CACHE.pop(f"/mem/{r['video']}", None)
    s = {}
    for k in ("pred_ev", "pred_argmax"):
        e = np.array([abs(o[k] - o["gt"]) / max(1, o["gt"]) for o in out])
        obo = np.array([abs(round(o[k]) - o["gt"]) <= 1 for o in out])
        s[k] = dict(MAE=float(e.mean()), OBO=float(obo.mean()))
    return dict(n=len(out), **s), out


def cmd_tb(a, M):
    """TemporalBench short_qa (binary caption choice). One prefix per video = 16 frames uniformly over the clip; all of
    that video's questions are blocks on it. Reports binary accuracy and multiple-binary accuracy (MBA: all questions of a
    video correct), overall and per source dataset / category. Videos: --tb_dir (unzipped short_video.zip)."""
    import re
    from collections import defaultdict
    qs = json.load(open(f"{a.tb_dir}/temporalbench_short_qa.json"))
    byv = defaultdict(list)
    for q in qs:
        byv[q["video_name"]].append(q)
    vids = sorted(byv)
    random.Random(0).shuffle(vids)
    if a.n:
        vids = vids[: a.n]
    out = []
    T = M[4].get("choice", 1.0)
    for v in vids:
        path = f"{a.tb_dir}/x/{v}"
        try:   # short clips: one sequential decode at 16/dur fps (16 seeks per video were ~10x slower)
            dur = probe(path)[0]
            fr = frames_at(path, 0.0, 16, max(16.0 / max(dur, 1e-3), 0.05))
            if len(fr) < 2:
                raise ValueError("no frames")
            dtg = dur / len(fr)
        except Exception as e:
            print("skip", v, repr(e)[:80]); continue
        if len(out) % 500 == 0:
            print("tb", len(out), "questions done", flush=True)
        blocks = []
        for q in byv[v]:
            lines = q["question"].split("\n")
            opts = [re.sub(r"^[A-Z]\.\s*", "", l).strip() for l in lines if re.match(r"^[A-Z]\.\s", l)]
            blocks.append(dict(type="choice", instr=lines[0].strip(), options=opts, task="tb", key=LETTER[q["GT"].strip()[0]]))
        rows = run_mem(M, [clip_rec(f"tb{len(out)}", fr, max(0.05, dtg), blocks, max_vis=96)])
        for r, q in zip(sorted(rows, key=lambda r: r["q_i"]), byv[v]):
            p = probs(r["logits"], T)
            out.append(dict(video=v, idx=q["idx"], dataset=q["dataset"], category=q["category"], key=LETTER[q["GT"].strip()[0]],
                            pred=int(np.argmax(p)), p=[round(float(x), 4) for x in p]))
        V._CACHE.clear()
    def summ(rr):
        acc = float(np.mean([o["pred"] == o["key"] for o in rr]))
        g = defaultdict(list)
        for o in rr:
            g[o["video"]].append(o["pred"] == o["key"])
        return dict(n=len(rr), binary_acc=acc, mba=float(np.mean([all(x) for x in g.values()])), n_videos=len(g))
    s = dict(all=summ(out))
    for k in ("dataset", "category"):
        for val in sorted({o[k] for o in out}):
            s[f"{k}:{val}"] = summ([o for o in out if o[k] == val])
    return s, out


VI_REPOS = {"v1_0": "Video-Index/Video-Index", "v1_1": "Video-Index/Video-Index-v1-1"}


def cmd_vi(a, M):
    """choose-only: (clips8) 8 uniform 4 s clips at 2 fps, choice head per clip, mean prob over clips;
    (global) 32 frames uniformly over the whole video in one prefix. argmin = how often the key is the LEAST likely."""
    from huggingface_hub import hf_hub_download
    it = [json.loads(l) for l in open(hf_hub_download(VI_REPOS[a.which], "items/test.jsonl", repo_type="dataset"))]
    if a.items:
        keep = set(a.items.split(","))
        it = [x for x in it if x["item_id"] in keep]
    if a.n:   # fixed random subset (seed 0) so every model sees the same items
        it = sorted(random.Random(0).sample(it, min(a.n, len(it))), key=lambda x: x["item_id"])
    out = []
    T = M[4].get("choice", 1.0)
    for x in it:
        try:
            path = hf_hub_download(VI_REPOS[a.which], x["video"], repo_type="dataset")
        except Exception as e:
            print("skip", x["item_id"], repr(e)[:100]); continue
        try:
            dur, W, H = probe(path)
        except Exception as e:
            print("skip", x["item_id"], "probe", repr(e)[:100]); continue
        q = dict(type="choice", instr=x["question"], options=x["options"], task="vi")
        recs = []
        for k in range(8):
            t0 = max(0.0, (k + 0.5) * dur / 8 - 2.0)
            fr = frames_at(path, t0, 8, 2.0)
            if len(fr) >= 2:
                recs.append(clip_rec(f"vi{x['item_id']}_{k}", fr, 0.5, [dict(q)], kind="clip"))
        try:
            fr, dtg = frames_uniform(path, 32)
        except Exception as e:
            print("skip", x["item_id"], repr(e)[:100]); continue
        if not any(r.get("kind") == "clip" for r in recs):
            print("skip", x["item_id"], "no clips"); continue
        recs.append(clip_rec(f"vi{x['item_id']}_g", fr, max(0.05, dtg), [dict(q)], max_vis=64, kind="global"))
        rows = run_mem(M, recs)
        pc = np.mean([probs(r["logits"], T) for r in rows if recs[r["rec_i"]]["kind"] == "clip"], 0)
        pg = [probs(r["logits"], T) for r in rows if recs[r["rec_i"]]["kind"] == "global"][0]
        key = x["answer_idx"]
        o = dict(item_id=x["item_id"], benchmark=x["benchmark"], group=x.get("capability_group"), key=key,
                 n_options=len(x["options"]), p_clips=[round(float(v), 4) for v in pc], p_global=[round(float(v), 4) for v in pg])
        for name, p in (("clips8", pc), ("global", pg), ("both", (pc + pg) / 2)):
            o[name] = dict(pred=int(np.argmax(p)), correct=bool(np.argmax(p) == key), argmin=bool(np.argmin(p) == key))
        out.append(o)
        print(x["item_id"], o["clips8"]["correct"], o["global"]["correct"], flush=True)
        for r in recs:
            V._CACHE.pop(f"/mem/{r['video']}", None)
    s = dict(n=len(out), chance=float(np.mean([1 / o["n_options"] for o in out])) if out else None)
    for name in ("clips8", "global", "both"):
        s[name] = dict(acc=float(np.mean([o[name]["correct"] for o in out])), argmin=float(np.mean([o[name]["argmin"] for o in out])),
                       n_correct=int(sum(o[name]["correct"] for o in out)))
    return s, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--data_root", default="/content")
    ap.add_argument("--gv_root", default="/content/gv")
    ap.add_argument("--recs", default="")
    ap.add_argument("--calib", default="")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--which", default="v1_1")
    ap.add_argument("--items", default="")
    ap.add_argument("--tb_dir", default="/content/tb")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--rows", default="")
    a = ap.parse_args()
    if a.cmd == "sta" and not a.n:
        a.n = 600
    if a.cmd == "resum":   # re-summarize saved typed rows (no model): --rows R.gz [--out]
        rows = [json.loads(l) for l in gzip.open(a.rows, "rt")]
        res = dict(raw=summarize(rows))
        print(json.dumps(res, indent=1)); json.dump(res, open(a.out, "w"), indent=1)
        return
    M = load(a)
    res, rows = dict(typed=cmd_typed, sta=cmd_sta, repcount=cmd_repcount, vi=cmd_vi, tb=cmd_tb)[a.cmd](a, M)
    res.update(cmd=a.cmd, ckpt=a.ckpt, base=a.base, which=a.which if a.cmd == "vi" else None)
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    if a.rows:
        with gzip.open(a.rows, "wt") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
