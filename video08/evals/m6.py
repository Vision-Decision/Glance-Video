"""M6: frozen Qwen3.5-0.8B zero-shot at fps=2 vs fps=30 (64 tokens / frame group), answers read at the end of the clip.

python -m video08.evals.m6 --bench {sta,presence,pushup} --arm {stream,native,recent4} --fps F --n N --out OUT.jsonl
  stream : our streaming engine, bounded KV (4 s full rate + 4-60 s at 2 fps, GDN state through everything)
  native : HF processor input (non-overlapping pairs, timestamps per pair), full context, offline - the pretrained format
  recent4: SimpleStream-style, only the last 4 frame groups, no history (fresh context per question time)
Benchmarks (all eval-only data; Charades = test split):
  sta      Charades-STA test, generative "start - end" -> mIoU, R@0.3/0.5/0.7
  presence Charades test whole-clip "did someone <action>?" (present classes vs same-count absent: half share an object or
           verb with a present class) -> AUROC, ECE, acc
  pushup   PushupBench, generative count -> exact, off-by-one, relative MAE
"""
import argparse, json, os, random, sys, time
import numpy as np
import torch

from video08.stream import Stream, load, yes_no_ids, p_yes
from video08 import vio, feed

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True); ap.add_argument("--arm", default="stream"); ap.add_argument("--fps", type=float, default=2)
ap.add_argument("--n", type=int, default=200); ap.add_argument("--out", required=True); ap.add_argument("--data", default=os.path.expanduser("~/glance-video-exp/data"))
ap.add_argument("--chunk", type=int, default=0, help="groups per LM forward (0: 0.5 s worth)"); ap.add_argument("--size", type=int, default=256)
ap.add_argument("--full_s", type=float, default=4.0); ap.add_argument("--sub_s", type=float, default=60.0)
ap.add_argument("--max_len", type=float, default=180.0, help="clip videos longer than this (s)")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
IDX = f"{a.data}/idx"
rng = random.Random(a.seed)
classes = json.load(open(f"{IDX}/classes.json"))
mapping = {l.split()[0]: l.split()[1:] for l in open(f"{a.data}/meta/ann/Charades/Charades_v1_mapping.txt")}


def phrase(c):
    p = classes[c]; return p[0].lower() + p[1:]


def items():
    if a.bench in ("sta", "presence"):
        rows = {r["id"]: r for r in map(json.loads, open(f"{IDX}/charades.jsonl"))}
        ev = [rows[i] for i in json.load(open(f"{IDX}/splits.json"))["eval"]]
        if a.bench == "sta":
            ev = [r for r in ev if r["sta"]]
        rng.shuffle(ev)
        for r in ev[: a.n]:
            if a.bench == "sta":
                qs = [dict(kind="gen", q=f"Give the query: '{s}', when does the described content occur in the video? "
                           f"Please return the start and end timestamps in seconds.", gt=[st, en], text=s) for s, st, en in r["sta"][:3]]
            else:
                pres = sorted({c for c, _, _ in r["actions"]})
                absent = [c for c in classes if c not in pres]
                objs = {mapping[c][0] for c in pres}; verbs = {mapping[c][1] for c in pres}
                hard = [c for c in absent if mapping[c][0] in objs or mapping[c][1] in verbs]
                k = min(len(pres), 6); P = rng.sample(pres, k)
                H = rng.sample(hard, min(len(hard), k // 2 + k % 2)); E = rng.sample([c for c in absent if c not in H], k - len(H))
                qs = [dict(kind="yn", q=f"Did someone {phrase(c)} at any point in the video? Answer yes or no.", y=int(c in pres),
                           cls=c, hard=int(c in H)) for c in P + H + E]
            yield dict(id=r["id"], path=r["path"], length=r["length"], qs=qs)
    elif a.bench == "pushup":
        rows = list(map(json.loads, open(f"{IDX}/pushup.jsonl"))); rng.shuffle(rows)
        for r in rows[: a.n]:
            yield dict(id=r["id"], path=r["path"], qs=[dict(kind="gen", q=f"How many repetitions of '{r['exercise']}' does "
                       f"the person complete in the video? Answer with a single number.", gt=r["count"])])


model, tok = load()
yid, nid = yes_no_ids(tok)
proc = None
if a.arm == "native":
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B")
st = Stream(model, tok, full_s=a.full_s, sub_s=a.sub_s, keep_all=a.arm == "native")
chunk = a.chunk or max(1, int(round(a.fps * 0.5)))
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["id"] for l in open(a.out)}
fo = open(a.out, "a")


@torch.no_grad()
def native_answer(frames, times, q, gen):
    if len(frames) % 2:
        frames = torch.cat([frames, frames[-1:]]); times = np.append(times, times[-1] + 1 / a.fps)
    msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    inp = proc(text=[text], videos=[frames.numpy()], return_tensors="pt", do_resize=False, do_sample_frames=False,
               video_metadata=[dict(fps=a.fps, total_num_frames=len(frames), frames_indices=list(range(len(frames))))]).to("cuda")
    if gen:
        o = model.generate(**inp, max_new_tokens=24, do_sample=False)
        return tok.decode(o[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return float(p_yes(model(**inp, logits_to_keep=1).logits[:, -1].float(), yid, nid)[0])


ap2 = None
BS = int(os.environ.get("GV_BATCH", "8"))


def dec(it):
    return vio.decode(it["path"], a.fps, a.size, 0.0, a.max_len)


def run_stream_batch(batch, decoded):
    """batched lockstep streams (same fps grid); each video answers at its own end"""
    vids, recs = [], []
    for it, (frames, times) in zip(batch, decoded):
        if len(frames) < 2:
            continue
        vids.append((frames, times)); recs.append(dict(id=it["id"], arm=a.arm, fps=a.fps, n_groups=len(frames), dur=float(times[-1]), res=[], it=it))
    if not vids:
        return []
    t0 = time.perf_counter()

    def on_end(i, row):
        rec = recs[i]; qs = rec["it"]["qs"]; rec.update(kv=st.kv_len(), pos=st.pos)
        yn = [q for q in qs if q["kind"] == "yn"]
        if yn:
            ps = p_yes(st.ask([q["q"] for q in yn], rows=[row] * len(yn))["logits"], yid, nid).tolist()
            rec["res"] += [dict(q, p=p) for q, p in zip(yn, ps)]
        for q in qs:
            if q["kind"] == "gen":
                rec["res"].append(dict(q, ans=st.generate(q["q"], max_new=24, row=row)))
    tm = feed.feed_batch(st, vids, chunk, on_end)
    wall = time.perf_counter() - t0
    for r in recs:
        r.pop("it"); r.update(wall_s=wall / len(recs), batch=len(recs), vit_s=tm["vit"] / len(recs), lm_s=tm["lm"] / len(recs))
    return recs


todo = [it for it in items() if it["id"] not in done]
if a.arm == "stream" and BS > 1:
    import concurrent.futures as cf
    todo.sort(key=lambda it: it.get("length", 0))  # similar lengths in one batch: few rows idle
    ex = cf.ThreadPoolExecutor(2); nxt = [ex.submit(dec, it) for it in todo[:BS]]
    for b in range(0, len(todo), BS):
        cur = [f.result() for f in nxt]
        nxt = [ex.submit(dec, it) for it in todo[b + BS:b + 2 * BS]]  # decode the next batch while this one runs
        for rec in run_stream_batch(todo[b:b + BS], cur):
            fo.write(json.dumps(rec) + "\n")
        fo.flush()
    print("done", a); raise SystemExit(0)

for it in todo:
    t0 = time.perf_counter()
    frames, times = vio.decode(it["path"], a.fps, a.size, 0.0, a.max_len)
    if len(frames) < 2:
        continue
    rec = dict(id=it["id"], arm=a.arm, fps=a.fps, n_groups=len(frames), dur=float(times[-1]), res=[])
    if a.arm == "native":
        for q in it["qs"]:
            out = native_answer(frames, times, q["q"], q["kind"] == "gen")
            rec["res"].append(dict(q, **({"ans": out} if q["kind"] == "gen" else {"p": out})))
    else:
        st.reset()
        if a.arm == "recent4":
            frames, times = frames[-4:], times[-4:]
        _, tm, _ = feed.feed(st, frames, times, chunk=chunk)
        rec.update(vit_s=tm["vit"], lm_s=tm["lm"], kv=st.kv_len(), pos=st.pos)
        yn = [q for q in it["qs"] if q["kind"] == "yn"]
        if yn:
            ps = p_yes(st.ask([q["q"] for q in yn])["logits"], yid, nid).tolist()
            rec["res"] += [dict(q, p=p) for q, p in zip(yn, ps)]
        for q in it["qs"]:
            if q["kind"] == "gen":
                rec["res"].append(dict(q, ans=st.generate(q["q"], max_new=24)))
    rec["wall_s"] = time.perf_counter() - t0
    fo.write(json.dumps(rec) + "\n"); fo.flush()
print("done", a)
