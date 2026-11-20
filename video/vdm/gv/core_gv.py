"""vdm v1 / Glance-Video backend: a clip is a NATIVE video prefix (Qwen3.5 video input, temporal patches of 2 frames,
"<t seconds>" timestamps), scored by the typed decision head of a Glance-Video checkpoint (dv/, configs/video_v0.json).

Unlike video/vdm/v1/core.py (Glance-4B-preview on a multi-image prefix, training-free), this model was trained on clips:
arrow of time, camera motion, speed, order, repetition counts, Charades events with temporal hard negatives.

Two ways to use it.

One-shot: clips and questions in one call.
  S = GVScorer("/content/out/gv0/ckpt_final.pt")         # or GVScorer(base=True) for zero-shot Qwen3.5-4B
  S.score(clips, "Does a car stops at the crossing?")                     -> [p_yes per clip]
  S.score_multi(clips, [dict(type="noul", instr=...), dict(type="choice", ...)]) -> [[result per question] per clip]

Scan: encode a video's clips once, then many queries.
  S.prepare("vid", clips, t0s=[...], dt=0.5)       # decode + vision tower once; cached on the GPU
  S.scan("vid", [dict(type="noul", instr=...)])    -> {"t0": [...], "results": [[p per question] per clip]}
  S.drop("vid")
This is the ClipBackend interface (prepare / scan / drop). A faster scan engine (one text query against many cached
clips in one forward, e.g. a cached prefix state per clip) can implement the same three methods and be passed to
server_gv.py instead of GVScorer. Here `scan` reuses the cached vision features and re-runs only the language model
(prefix + question blocks), i.e. the vision tower and video decoding are paid once per video.

clips: list of uint8 arrays [T,H,W,3] (or lists of PIL frames), frames in time order, `dt` seconds apart.
"""
import os, sys
import numpy as np
import torch

ROOT = os.environ.get("DV_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from dv.config import Config  # noqa: E402
from dv.data import video as V  # noqa: E402
from dv.data.mixed import MixedDataset, collate_mixed, QTYPES  # noqa: E402
from dv.model.build import build, load_ckpt  # noqa: E402
from dv.model.typed import typed_logits  # noqa: E402

YESNO, CHOICE = "yesno", "choice"


class ClipBackend:
    """What server_gv.py needs from a backend."""

    def prepare(self, video_id, clips, t0s=None, dt=0.5):
        raise NotImplementedError

    def scan(self, video_id, questions):
        """-> {"t0": [clip start times], "results": [[p_yes | option probs] per question] per clip]}"""
        raise NotImplementedError

    def drop(self, video_id):
        raise NotImplementedError


def _with_target(q):
    if "target" in q:
        return dict(q)
    return dict(q, target=0.5 if q["type"] == "noul" else [1.0 / len(q["options"])] * len(q["options"]))


class GVScorer(ClipBackend):
    def __init__(self, ckpt="", base=False, device="cuda", max_vis=96, temps=None, chunk=16):
        if base:
            cfg = Config().update(dict(task="v1", attn_mask="causal", lora_r=0))
            sd = None
        else:
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)
            cfg = Config().update(sd["cfg"])
        cfg.data_root = "/mem"
        self.net, self.heads, self.proc = build(cfg, device, lora=not base)
        if not base:
            load_ckpt(ckpt, self.net, self.heads)
        self.net.eval(); self.heads.eval()
        self.cfg, self.max_vis, self.dev, self.chunk = cfg, max_vis, device, chunk
        self.temps = temps or ((sd or {}).get("extra", {}).get("temps") or {})
        self.ds = MixedDataset(None, "/mem", self.proc, cfg, train=False, recs=[])
        self.cache = {}          # video_id -> dict(t0=[...], chunks=[(items_without_questions, feats)])
        self._n = 0

    # ------------------------------------------------------------------ helpers
    def _arr(self, clip):
        if isinstance(clip, np.ndarray):
            return clip
        return np.stack([np.asarray(f.convert("RGB")) for f in clip])

    def _item(self, clip, questions, dt):
        a = self._arr(clip)
        self._n += 1
        k = f"c{self._n}"
        V._CACHE[f"/mem/{k}"] = a
        self.ds.recs = [dict(video=k, frames=list(range(len(a))), dt=dt, max_vis=self.max_vis,
                             questions=[_with_target(q) for q in questions])]
        it = self.ds[0]
        V._CACHE.pop(f"/mem/{k}", None)
        return it

    def _q_fields(self, questions):
        """question part of a dataset item (same as MixedDataset.__getitem__ for typed questions)"""
        tok = self.ds.tok
        f = dict(q_ids=[], qtype=[], labs=[], tgts=[], present=[], has_box=[], box=[], points=[], qs=[])
        for q in questions:
            q = _with_target(q)
            ids, lab = tok.typed_block(q)
            f["q_ids"].append(ids); f["qtype"].append(QTYPES[q["type"]]); f["labs"].append(lab)
            f["tgts"].append([1 - q["target"], q["target"]] if q["type"] == "noul" else list(q["target"]))
            f["present"].append(False); f["has_box"].append(False); f["box"].append([0, 0, 0, 0]); f["points"].append([])
            f["qs"].append(q)
        return f

    @torch.no_grad()
    def _forward(self, items, feats, questions):
        batch = collate_mixed(items, self.ds.tok)
        batch.pop("meta", None)
        batch.pop("pixel_values_videos", None)   # vision features are passed in (cached)
        b = {k: (v.to(self.dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hp, hq = self.net(b, feats)
        M = hq.shape[0]
        lab = self.heads.typed(hq[torch.arange(M, device=hq.device), b["q_last"]])
        tl = typed_logits(lab, b["lab_idx"]).float().cpu()
        out, m = [], 0
        for _ in items:
            row = []
            for q in questions:
                lg = tl[m][tl[m] > -1e30]
                T = self.temps.get("noul" if q["type"] == "noul" else "choice", 1.0)
                p = torch.softmax(lg / T, -1)
                row.append(float(p[1]) if q["type"] == "noul" else [float(v) for v in p])
                m += 1
            out.append(row)
        return out

    def _encode(self, items):
        pv = torch.cat([it["pixel_values_videos"] for it in items]).to(self.dev)
        grid = torch.stack([it["video_grid_thw"] for it in items]).to(self.dev)
        return self.net.video_features(pv, grid)

    # ------------------------------------------------------------------ scan interface
    @torch.no_grad()
    def prepare(self, video_id, clips, t0s=None, dt=0.5):
        dummy = [dict(type="noul", instr="?")]
        chunks = []
        for i in range(0, len(clips), self.chunk):
            items = [self._item(c, dummy, dt) for c in clips[i:i + self.chunk]]
            feats = self._encode(items)
            for it in items:
                it.pop("pixel_values_videos", None)      # keep only what the LM needs
            chunks.append((items, feats))
        self.cache[video_id] = dict(t0=list(t0s) if t0s is not None else list(range(len(clips))), chunks=chunks, dt=dt)
        return dict(video_id=video_id, n_clips=len(clips))

    @torch.no_grad()
    def scan(self, video_id, questions):
        c = self.cache[video_id]
        qf = self._q_fields(questions)
        res = []
        for items, feats in c["chunks"]:
            its = [dict(it, **qf) for it in items]
            res += self._forward(its, feats, questions)
        return dict(t0=c["t0"], results=res)

    def drop(self, video_id):
        self.cache.pop(video_id, None)

    # ------------------------------------------------------------------ one-shot
    @torch.no_grad()
    def score_multi(self, clips, questions, dt=0.5, batch=16):
        """-> per clip, per question: p_yes (noul) or option-probability list (choice)."""
        out = []
        for i in range(0, len(clips), batch):
            items = [self._item(c, questions, dt) for c in clips[i:i + batch]]
            feats = self._encode(items)
            out += self._forward(items, feats, questions)
        return out

    def score(self, clips, question, kind=YESNO, options=None, dt=0.5, batch=16):
        q = dict(type="noul", instr=question) if kind == YESNO else dict(type="choice", instr=question, options=list(options))
        return [r[0] for r in self.score_multi(clips, [q], dt=dt, batch=batch)]
