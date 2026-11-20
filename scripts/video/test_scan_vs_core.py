"""GPU parity: scan engine (scan/engine.py, the scan_server.py backend) vs video/vdm/gv/core_gv.py on the same clips.

  python scripts/video/test_scan_vs_core.py --ckpt /content/ckpt/glance-video-v0/ckpt_final.pt --n_videos 6 \
      --out /content/out/scan/gv_scan_vs_core.json
Clips: Charades-STA TEST videos (held out from Glance-Video training), 4 frames at 2 fps (2 s), stride 1 s, from the
normalised 8-fps clips in /content/gv/clips/charades. Questions: the video's STA sentences ("Does this clip show: ...?").
Compares P(yes) of (a) core_gv.score_multi (one forward per clip batch), (b) core_gv.prepare/scan (cached vision
features), (c) scan engine encode/query (cached prefix state, LoRA merged, fused ops). Also writes the per-clip P(t) of
every query with its ground-truth moment, for validating scan P(t) curves.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "video/vdm/gv"))
import numpy as np
import torch

from dv.data import video as V

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--n_videos", type=int, default=6)
ap.add_argument("--sta", default="/content/gv/raw/ann/charades_sta_test.txt")
ap.add_argument("--clips_dir", default="/content/gv/clips/charades")
ap.add_argument("--out", default="/content/out/scan/gv_scan_vs_core.json")
a = ap.parse_args()

byv = {}
for l in open(a.sta):
    head, s = l.strip().split("##")
    v, t0, t1 = head.split()
    byv.setdefault(v, []).append((float(t0), float(t1), s.strip().rstrip(".")))
vids = [v for v in sorted(byv) if os.path.exists(f"{a.clips_dir}/{v}.mp4")]
rng = np.random.default_rng(0)
vids = list(rng.choice(vids, size=min(a.n_videos, len(vids)), replace=False))

videos = []
for v in vids:
    arr = V.decode_clip(f"{a.clips_dir}/{v}.mp4")          # 8 fps
    dur = len(arr) / 8.0
    t0s = np.arange(0, max(0.0, dur - 2.0) + 1e-6, 1.0)
    clips = [arr[int(t * 8): int(t * 8) + 16: 4] for t in t0s]
    keep = [i for i, c in enumerate(clips) if len(c) == 4]
    videos.append(dict(vid=v, dur=dur, t0=[float(t0s[i]) for i in keep], clips=[clips[i] for i in keep],
                       qs=[dict(type="noul", instr=f"Does this clip show: {s}?") for _, _, s in byv[v]], sta=byv[v]))

from core_gv import GVScorer  # noqa: E402
S = GVScorer(a.ckpt, max_vis=96)
res_core, res_cached, t_core, t_scan_core = [], [], 0.0, 0.0
for vd in videos:
    t = time.time(); res_core.append(S.score_multi(vd["clips"], vd["qs"], dt=0.5)); t_core += time.time() - t
    S.prepare(vd["vid"], vd["clips"], t0s=vd["t0"], dt=0.5)
    t = time.time(); res_cached.append(S.scan(vd["vid"], vd["qs"])["results"]); t_scan_core += time.time() - t
    S.drop(vd["vid"])
del S; torch.cuda.empty_cache()

from scan.pilot_charades import load_glance  # noqa: E402
eng, _ = load_glance(a.ckpt)
eng.dt, eng.clip_max_vis = 0.5, 96
res_scan, t_enc, t_q = [], 0.0, 0.0
with torch.no_grad():
    for vd in videos:
        t = time.time(); bank = eng.encode(vd["clips"], chunk=64); t_enc += time.time() - t
        per_q = []
        for q in vd["qs"]:
            t = time.time(); tl = eng.query(bank, q)["typed_logits"]; t_q += time.time() - t
            per_q.append(torch.softmax(tl[:, :2].float(), -1)[:, 1].numpy())
        res_scan.append(np.stack(per_q, 1))                       # [clips, queries]
        eng.free_graphs(bank) if hasattr(eng, "free_graphs") else None
        del bank

core = np.concatenate([np.array(r).ravel() for r in res_core])          # per video [clips, queries] -> flat
cached = np.concatenate([np.array(r).ravel() for r in res_cached])
scan = np.concatenate([np.asarray(r).ravel() for r in res_scan])


def cmp(x, y):
    lx, ly = np.log(x / (1 - x + 1e-9) + 1e-9), np.log(y / (1 - y + 1e-9) + 1e-9)
    return dict(n=len(x), mean_abs_dP=float(np.abs(x - y).mean()), max_abs_dP=float(np.abs(x - y).max()),
                decision_agree=float(((x > 0.5) == (y > 0.5)).mean()), logit_corr=float(np.corrcoef(lx, ly)[0, 1]))


out = dict(ckpt=a.ckpt, n_videos=len(videos), n_clips=int(sum(len(v["clips"]) for v in videos)),
           core_vs_cachedcore=cmp(core, cached), core_vs_scan=cmp(core, scan),
           sec=dict(core_score_multi=t_core, core_cached_scan=t_scan_core, scan_encode=t_enc, scan_queries=t_q,
                    n_queries=int(sum(len(v["qs"]) for v in videos))),
           curves=[dict(vid=vd["vid"], dur=vd["dur"], clip_s=2.0, fps=2.0, stride_s=1.0, t0=vd["t0"],
                        queries=[dict(sentence=s, gt=[g0, g1],
                                      p_core=[round(float(r[qi]), 4) for r in res_core[k]],
                                      p_scan=[round(float(x), 4) for x in res_scan[k][:, qi]])
                                 for qi, (g0, g1, s) in enumerate(vd["sta"])])
                   for k, vd in enumerate(videos)])
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "curves"}, indent=1))
