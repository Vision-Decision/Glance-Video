"""vdm v1 core: score SHORT CLIPS (several frames) with yes/no or multiple-choice questions.

One Glance-4B-preview model serves two backends over the identical clip input:
  glance  - LoRA on, Glance's typed yes/no / choice head (calibrated temperatures), block-bidirectional question block
  qwen    - LoRA disabled (= base Qwen3.5-4B), causal question block, LM-head next-token readout
            (P(yes) = softmax(logsumexp{yes,Yes}, logsumexp{no,No}); choice = softmax over letter tokens)
Glance only supports single images, so a clip enters as a multi-image prefix: <vision_start><image_pad>*n<vision_end>
per frame, frames in time order, consecutive rotary positions (each frame advances the position by max(gh, gw)).
Optional third backend `qwen-video`: the base model through the HF processor's native video input.
"""
import math
from contextlib import contextmanager

import torch
from PIL import Image

from glance import Glance
from glance.prompt import resize_dims
from glance.backbone import build_masks  # noqa: F401  (kept for reference)

YESNO = "yesno"
CHOICE = "choice"


class ClipScorer:
    def __init__(self, repo="Vision-Decision/Glance-4B-preview", device="cuda", max_vis_tokens=256):
        self.g = Glance.from_pretrained(repo, device=device)
        self.tok = self.g.tok
        self.mvt = max_vis_tokens
        hf_tok = self.g.processor.tokenizer
        ids = lambda w: hf_tok(w, add_special_tokens=False)["input_ids"]
        self.yes_ids = sorted({ids(w)[0] for w in ("yes", "Yes")})
        self.no_ids = sorted({ids(w)[0] for w in ("no", "No")})
        self.letter_ids = [ids(chr(65 + i))[0] for i in range(26)]
        from peft.tuners.tuners_utils import BaseTunerLayer
        self._lora = [m for m in self.g.net.hf.modules() if isinstance(m, BaseTunerLayer)]

    # ------------------------------------------------------------------ backend switches
    @contextmanager
    def backend(self, name):
        net = self.g.net
        old_mask = net.attn_mask
        if name == "qwen":
            for m in self._lora:
                m.enable_adapters(False)
            net.attn_mask = "causal"
        try:
            yield
        finally:
            for m in self._lora:
                m.enable_adapters(True)
            net.attn_mask = old_mask

    # ------------------------------------------------------------------ batching
    def _items(self, clips, kind, question, options):
        """clips: list of lists of PIL frames (all frames of all clips the same size)."""
        tok, ip = self.tok, self.g.processor.image_processor
        if kind == YESNO:
            q_ids, lab = tok.typed_block("yesno", question)
        else:
            q_ids, lab = tok.typed_block("choice", question, list(options))
        W, H = clips[0][0].size
        w, h = resize_dims(W, H, self.mvt, self.g.cfg["max_upscale"])
        items = []
        for frames in clips:
            pvs, grids, segs = [], [], [("ids", tok.user_open)]
            for k, fr in enumerate(frames):
                pv = ip(images=[fr.convert("RGB").resize((w, h), Image.BICUBIC)], do_resize=False, return_tensors="pt")
                pvs.append(pv["pixel_values"])
                grids.append(pv["image_grid_thw"][0])
                segs += [("ids", [tok.vs]), ("img", k), ("ids", [tok.ve])]
            items.append(dict(pvs=pvs, grids=grids, segs=segs, q_ids=[q_ids]))
        return items, lab

    def _collate(self, items, merge=2, pad_to=64):
        tok = self.tok
        B = len(items)
        g0 = items[0]["grids"][0]
        gh, gw = int(g0[1] // merge), int(g0[2] // merge)
        seq = [sum(len(s[1]) if s[0] == "ids" else gh * gw for s in it["segs"]) for it in items]
        Lp = -(-max(seq) // pad_to) * pad_to
        p_ids = torch.full((B, Lp), tok.pad, dtype=torch.long)
        p_valid = torch.zeros(B, Lp, dtype=torch.bool)
        p_img = torch.zeros(B, Lp, dtype=torch.bool)
        p_pos = torch.zeros(3, B, Lp, dtype=torch.long)
        q_start = []
        ii, jj = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
        for b, it in enumerate(items):
            s = Lp - seq[b]
            p_valid[b, s:] = True
            cur = 0
            for kind, val in it["segs"]:
                if kind == "ids":
                    n = len(val)
                    p_ids[b, s:s + n] = torch.tensor(val, dtype=torch.long)
                    p_pos[:, b, s:s + n] = cur + torch.arange(n)
                    s += n
                    cur += n
                else:
                    n = gh * gw
                    p_ids[b, s:s + n] = tok.pad_img
                    p_img[b, s:s + n] = True
                    p_pos[0, b, s:s + n] = cur
                    p_pos[1, b, s:s + n] = cur + ii.flatten()
                    p_pos[2, b, s:s + n] = cur + jj.flatten()
                    s += n
                    cur += max(gh, gw)
            q_start.append(cur)
        Q = [it["q_ids"][0] for it in items]
        Tq = -(-max(len(q) for q in Q) // 16) * 16
        q_ids = torch.full((B, Tq), tok.pad, dtype=torch.long)
        q_valid = torch.zeros(B, Tq, dtype=torch.bool)
        q_last = torch.zeros(B, dtype=torch.long)
        q_pos = torch.zeros(3, B, Tq, dtype=torch.long)
        for m, q in enumerate(Q):
            q_ids[m, :len(q)] = torch.tensor(q)
            q_valid[m, :len(q)] = True
            q_last[m] = len(q) - 1
            q_pos[:, m, :] = q_start[m] + torch.arange(Tq)
        pv = torch.cat([p for it in items for p in it["pvs"]])
        gt = torch.stack([g for it in items for g in it["grids"]])
        return dict(pixel_values=pv, grid_thw=gt, p_ids=p_ids, p_valid=p_valid, p_img_mask=p_img, p_pos=p_pos,
                    q_ids=q_ids, q_valid=q_valid, q_last=q_last, q_pos=q_pos, q_img=torch.arange(B))

    @torch.no_grad()
    def _hidden(self, batch):
        g = self.g
        dev = g.device
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        feats = g.net.image_features(b["pixel_values"].to(g.dtype), b["grid_thw"])
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if g._autocast else torch.no_grad()
        with ctx:
            hp, hq = g.net(b, feats)
        M = hq.shape[0]
        return hq[torch.arange(M, device=hq.device), b["q_last"]]

    # ------------------------------------------------------------------ public
    @torch.no_grad()
    def score(self, clips, question, backend="glance", kind=YESNO, options=None, batch=16):
        """-> list of P(yes) (yesno) or list of option-probability lists (choice), one per clip."""
        out = []
        T = self.g.temperatures
        for i in range(0, len(clips), batch):
            items, lab = self._items(clips[i:i + batch], kind, question, options)
            with self.backend(backend):
                h = self._hidden(self._collate(items))
                if backend == "glance":
                    lg = self.g.heads.typed(h).float()[:, lab]
                    p = torch.softmax(lg / T.get("noul" if kind == YESNO else "choice", 1.0), -1)
                else:
                    logits = self.g.net.hf.lm_head(h.to(self.g.net.hf.lm_head.weight.dtype)).float()
                    if kind == YESNO:
                        two = torch.stack([torch.logsumexp(logits[:, self.no_ids], -1),
                                           torch.logsumexp(logits[:, self.yes_ids], -1)], -1)
                        p = torch.softmax(two, -1)
                    else:
                        p = torch.softmax(logits[:, self.letter_ids[:len(options)]], -1)
            p = p.cpu()
            out += [float(r[1]) for r in p] if kind == YESNO else [[float(v) for v in r] for r in p]
        return out


# ---------------------------------------------------------------------- curve -> count
def segments(ts, ps, hi=0.5, lo=0.35, merge_gap=2.0):
    """Hysteresis segments of P(t) (t = clip centers), merged when closer than merge_gap seconds."""
    segs, on, start = [], False, None
    for t, p in zip(ts, ps):
        if not on and p >= hi:
            on, start = True, t
        elif on and p < lo:
            segs.append([start, prev])
            on = False
        prev = t
    if on:
        segs.append([start, ts[-1]])
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < merge_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return merged


def count_distribution(ts, ps):
    dist = {}
    for hi in (0.3, 0.5, 0.7):
        for gap in (1.0, 2.0, 4.0):
            dist[f"hi{hi}_gap{gap:g}"] = len(segments(ts, ps, hi, hi - 0.15, gap))
    return dist
