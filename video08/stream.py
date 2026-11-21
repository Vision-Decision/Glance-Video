"""Streaming Qwen3.5 (GDN + full attention 3:1) for 30 fps video: one forward per frame group, bounded full-attention KV,
GDN state carried through the whole stream, question blocks forked from a snapshot and never written back.

Token layout of one frame group (Qwen3.5 video format, one temporal patch = 2 frames):
    [<t.t seconds>]  <|vision_start|>  64 x <video embeds>  <|vision_end|>
The text timestamp is inserted at most every `ts_every` seconds (0.5 s by default). A group at sampling rate `fps` is the
sliding pair (frame at t - 1/fps, frame at t) through the Conv3d tubelet (step 1, no waiting for a second frame).

KV policy (full-attention layers only; GDN layers keep their fixed-size state, nothing is evicted there):
    sink (prompt header) + every group from the last `full_s` seconds + groups in [now - sub_s, now - full_s) that sit
    on the `sub_fps` grid. Older groups live only in the GDN state.
Positions: MRoPE as Qwen3.5 assigns them (t/h/w = base + (0, row, col), base advances by max(gh, gw) per group). When the
next position would pass `max_pos`, every retained key after the sink is rotated back by the gap between the sink and the
oldest retained group ("continuous RoPE"), so relative positions inside the window are unchanged and positions stay in
the trained range for hour-long streams.
"""
import copy, math, time, types
import torch
import torch.nn.functional as F

IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def load(model_id="Qwen/Qwen3.5-0.8B", device="cuda", dtype=torch.bfloat16, attn=None):
    import os
    attn = attn or os.environ.get("GV_ATTN", "sdpa")
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if attn == "split":
        from .attn import register
        register()
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype, attn_implementation=attn).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def patchify(frames, patch=16, tp=2, merge=2):
    """frames: (G, 2, 3, H, W) float in [0,1] (pairs) -> pixel rows (G*gh*gw, 3*tp*patch*patch), grid (G,3) in the order
    Qwen2-VL's image processor emits (merge-window major)."""
    G, T, C, H, W = frames.shape
    x = (frames - 0.5) / 0.5
    gh, gw = H // patch, W // patch
    x = x.reshape(G, 1, tp, C, gh // merge, merge, patch, gw // merge, merge, patch)
    x = x.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    x = x.reshape(G * gh * gw, C * tp * patch * patch)
    grid = torch.tensor([[1, gh, gw]] * G, device=frames.device)
    return x, grid


def _assign_state(self, recurrent_states, state_idx=0, **kw):
    self.recurrent_states[state_idx] = recurrent_states
    return recurrent_states


class Stream:
    def __init__(self, model, tok, full_s=4.0, sub_s=60.0, sub_fps=2.0, ts_every=0.5, max_pos=120_000, keep_all=False,
                 readout_layers=(), system=None):
        self.m, self.tok = model, tok
        self.lm = model.model.language_model
        self.visual = model.model.visual
        self.emb = self.lm.embed_tokens
        self.cfg = model.config
        self.dev = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.full_s, self.sub_s, self.sub_fps, self.ts_every, self.max_pos = full_s, sub_s, sub_fps, ts_every, max_pos
        self.keep_all = keep_all  # unbounded KV (reference "full KV" arm)
        self.readout_layers = tuple(readout_layers)
        self.ids = {k: tok.convert_tokens_to_ids(k) for k in ("<|vision_start|>", "<|vision_end|>", "<|video_pad|>")}
        rot = self.lm.rotary_emb
        self.inv_freq = rot.inv_freq.float().to(self.dev)
        self.full_layers = [i for i, t in enumerate(self.cfg.text_config.layer_types) if t == "full_attention"]
        self.system = system
        import os
        self.split_ok = os.environ.get("GV_SPLIT", "1") == "1" and torch.cuda.is_available()
        if self.split_ok:
            from .attn import register
            register()
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self, zero_state=False, batch=1):
        """batch > 1: B streams in lockstep on the same time grid (identical token layout, so they share one forward;
        rows that finish early are dropped with drop())."""
        from transformers.cache_utils import DynamicCache
        self.B = batch
        self.cache = DynamicCache(config=self.cfg.text_config)
        self.pos = 0  # next position id
        self.meta_t = torch.empty(0, device=self.dev)  # per cached token: group time (-1 = sink / timestamp-free text)
        self.meta_pos = torch.empty(0, dtype=torch.long, device=self.dev)
        self.last_ts = -1e9
        self.now = 0.0
        self.n_written = 0
        self.rebases = 0
        head = (f"{IM_START}system\n{self.system}{IM_END}\n" if self.system else "") + f"{IM_START}user\n<|vision_start|>"  # Qwen3.5 wraps the whole video once more
        self._text(head, sink=True)

    def _forward(self, embeds, pos3, cache=None, want_hidden=False):
        cache = self.cache if cache is None else cache
        with self.fast_attn():
            out = self.lm(inputs_embeds=embeds, position_ids=pos3, past_key_values=cache, use_cache=True,
                          output_hidden_states=bool(self.readout_layers) or want_hidden)
        return out

    def fast_attn(self):
        """no-grad paths (frame writes, generative asks) use the exact mask-free split attention: SDPA with a boolean
        mask falls back to a slow kernel. Readouts that need gradients keep the model's own implementation."""
        import contextlib
        cfg = self.lm.config
        if not self.split_ok or torch.is_grad_enabled():
            return contextlib.nullcontext()

        @contextlib.contextmanager
        def ctx():
            old = cfg._attn_implementation; cfg._attn_implementation = "split"
            try:
                yield
            finally:
                cfg._attn_implementation = old
        return ctx()

    def _text(self, s, sink=False, t=-1.0):
        ids = torch.tensor([self.tok.encode(s, add_special_tokens=False)], device=self.dev)
        n = ids.shape[1]
        pos = torch.arange(self.pos, self.pos + n, device=self.dev)
        self._forward(self.emb(ids).expand(self.B, -1, -1), pos.view(1, 1, -1).expand(3, self.B, -1))
        self.pos += n
        self.meta_t = torch.cat([self.meta_t, torch.full((n,), -1.0 if sink else t, device=self.dev)])
        self.meta_pos = torch.cat([self.meta_pos, pos])

    # ------------------------------------------------------------------ vision
    @torch.no_grad()
    def encode(self, pairs):
        """pairs (G, 2, 3, H, W) in [0,1] on device -> (G, ntok, D) LM-space embeddings, grid (gh, gw) after merge"""
        px, grid = patchify(pairs.to(self.dev, torch.float32))
        out = self.visual(px.to(self.dtype), grid_thw=grid)
        emb = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out
        if isinstance(emb, (tuple, list)):
            emb = torch.cat(list(emb), 0)
        G = pairs.shape[0]
        gh, gw = pairs.shape[-2] // 32, pairs.shape[-1] // 32
        return emb.reshape(G, gh * gw, -1), (gh, gw)

    # ------------------------------------------------------------------ write
    @torch.no_grad()
    def write(self, vis, times, grid, return_feats=False):
        """Write G frame groups (vis: (G, ntok, D) embeddings, or (B, G, ntok, D) for a batched stream; times: list of G
        seconds) in ONE forward (G=1 is the strict per-frame mode). Returns per-group pooled hidden states
        {layer: (G, D)} (or (G, B, D) when batched) if return_feats."""
        gh, gw = grid
        batched = vis.dim() == 4
        if not batched:
            vis = vis[None]
        if self.pos >= self.max_pos and not self.keep_all:
            self._rebase()
        embs, poss, mt, mp, spans = [], [], [], [], []
        p = self.pos
        vs, ve = self.emb.weight[self.ids["<|vision_start|>"]], self.emb.weight[self.ids["<|vision_end|>"]]
        for g, t in enumerate(times):
            if t - self.last_ts >= self.ts_every - 1e-6:
                ids = self.tok.encode(f"<{t:.1f} seconds>", add_special_tokens=False)
                e = self.emb(torch.tensor(ids, device=self.dev))
                embs.append(e); n = len(ids)
                poss.append(torch.arange(p, p + n, device=self.dev).view(1, -1).expand(3, -1)); p += n
                mt.append(torch.full((n,), float(t), device=self.dev)); self.last_ts = t
            embs.append(vs[None]); poss.append(torch.full((3, 1), p, device=self.dev)); p += 1
            r = torch.arange(gh, device=self.dev).repeat_interleave(gw); c = torch.arange(gw, device=self.dev).repeat(gh)
            s0 = sum(e.shape[0] for e in embs)
            embs.append(torch.zeros(gh * gw, vis.shape[-1], device=self.dev, dtype=vis.dtype)); poss.append(torch.stack([torch.full_like(r, p), p + r, p + c])); p += max(gh, gw)
            spans.append((s0, s0 + gh * gw))
            embs.append(ve[None]); poss.append(torch.full((3, 1), p, device=self.dev)); p += 1
            mt.append(torch.full((gh * gw + 2,), float(t), device=self.dev))
        E = torch.cat(embs, 0)[None].to(self.dtype).expand(self.B, -1, -1).clone()
        for g, (a, b) in enumerate(spans):
            E[:, a:b] = vis[:, g].to(self.dtype)
        P = torch.cat(poss, 1)[:, None].expand(3, self.B, -1)
        out = self._forward(E, P, want_hidden=return_feats)
        self.meta_t = torch.cat([self.meta_t] + mt)
        self.meta_pos = torch.cat([self.meta_pos, P[0, 0]])
        self.pos = p
        self.now = times[-1]; self.n_written += len(times)
        feats = None
        if return_feats:
            hs = out.hidden_states
            layers = self.readout_layers or (len(hs) - 1,)
            feats = {L: torch.stack([hs[L][:, a:b].float().mean(1) for a, b in spans]) for L in layers}  # (G, B, D)
            if not batched:
                feats = {L: v[:, 0] for L, v in feats.items()}
        if not self.keep_all:
            self.evict()
        return feats

    def kv_len(self):
        return self.cache.layers[self.full_layers[0]].keys.shape[-2]

    def evict(self):
        t = self.meta_t; now = self.now
        keep = (t < 0) | (t >= now - self.full_s)
        mid = (t >= now - self.sub_s) & (t < now - self.full_s)
        # 4-60 s: keep the first group of every 1/sub_fps-second bin (independent of where the stream's grid starts;
        # an on-grid test on absolute time silently kept nothing when concatenated clips shifted the grid)
        b = torch.floor(t * self.sub_fps + 1e-6)
        first = torch.full_like(t, float("inf")).scatter_reduce(0, (b - b.min()).long().clamp(min=0), t, "amin", include_self=True) \
            if t.numel() else t
        on_grid = t == first[(b - b.min()).long().clamp(min=0)]
        keep = keep | (mid & on_grid)
        if bool(keep.all()):
            return
        idx = keep.nonzero().squeeze(1)
        for i in self.full_layers:
            L = self.cache.layers[i]
            L.keys = L.keys.index_select(-2, idx); L.values = L.values.index_select(-2, idx)
        self.meta_t, self.meta_pos = t[idx], self.meta_pos[idx]

    def _rotate_keys(self, shift, sel):
        """rotate cached keys at token mask `sel` by -shift positions (uniform shift of t,h,w == rotation of every freq)"""
        ang = -shift * self.inv_freq
        cos = torch.cat([ang.cos(), ang.cos()]); sin = torch.cat([ang.sin(), ang.sin()])
        rd = cos.shape[0]
        for i in self.full_layers:
            L = self.cache.layers[i]
            k = L.keys[:, :, sel].float()
            kr, kp = k[..., :rd], k[..., rd:]
            x1, x2 = kr[..., : rd // 2], kr[..., rd // 2:]
            rot = torch.cat([-x2, x1], -1)
            kr = kr * cos + rot * sin
            L.keys[:, :, sel] = torch.cat([kr, kp], -1).to(L.keys.dtype)

    def _rebase(self):
        sink = self.meta_t < 0
        sink_end = int(self.meta_pos[sink].max()) + 1 if bool(sink.any()) else 0
        oldest = int(self.meta_pos[~sink].min()) if bool((~sink).any()) else self.pos
        shift = oldest - sink_end
        if shift <= 0:
            return
        self._rotate_keys(shift, ~sink)
        self.meta_pos = torch.where(sink, self.meta_pos, self.meta_pos - shift)
        self.pos -= shift
        self.rebases += 1

    # ------------------------------------------------------------------ read
    def fork(self, n, rows=None):
        """copy of the cache with n rows; row i is stream rows[i] (default: all from stream 0)"""
        rows = torch.zeros(n, dtype=torch.long, device=self.dev) if rows is None else torch.as_tensor(rows, device=self.dev)
        new = copy.copy(self.cache); new.layers = []
        for L in self.cache.layers:
            M = copy.copy(L)
            if isinstance(getattr(L, "keys", None), torch.Tensor) and L.keys.numel():
                M.keys = L.keys.index_select(0, rows); M.values = L.values.index_select(0, rows)
            for a in ("conv_states", "recurrent_states"):
                if hasattr(L, a):
                    setattr(M, a, {k: (v.index_select(0, rows) if v is not None else None) for k, v in getattr(L, a).items()})
            for a in ("is_conv_states_initialized", "is_recurrent_states_initialized", "has_previous_state", "conv_kernel_size"):
                if hasattr(L, a):
                    setattr(M, a, dict(getattr(L, a)))
            if hasattr(L, "recurrent_states"):
                # assign instead of the in-place copy_: the readout forward may need the old state for backward
                M.update_recurrent_state = types.MethodType(_assign_state, M)
            new.layers.append(M)
        return new

    def drop(self, keep):
        """keep only stream rows `keep` (batched mode)"""
        keep = torch.as_tensor(keep, device=self.dev)
        for L in self.cache.layers:
            if isinstance(getattr(L, "keys", None), torch.Tensor) and L.keys.numel():
                L.keys = L.keys.index_select(0, keep); L.values = L.values.index_select(0, keep)
            for a in ("conv_states", "recurrent_states"):
                if hasattr(L, a):
                    d = getattr(L, a)
                    for k, v in d.items():
                        if v is not None:
                            d[k] = v.index_select(0, keep).contiguous()
        self.B = len(keep)

    def question_ids(self, q, suffix=""):
        s = f"<|vision_end|>{q}{IM_END}\n{IM_START}assistant\n<think>\n\n</think>\n\n{suffix}"
        return self.tok.encode(s, add_special_tokens=False)

    @torch.no_grad()
    def ask(self, questions, extra_embeds=None, answer_tokens=None, hidden_layers=None, rows=None):
        """N question blocks forked from the current snapshot; nothing is written back.
        Returns dict(logits=(N,V) at the last question token, hidden={layer: (N, L_extra+1, D)} for readout heads)."""
        N = len(questions)
        ids = [self.question_ids(q) for q in questions]
        Lmax = max(map(len, ids))
        cache = self.fork(N, rows)
        E = torch.zeros(N, Lmax, self.cfg.text_config.hidden_size, device=self.dev, dtype=self.dtype)
        last = torch.tensor([len(x) - 1 for x in ids], device=self.dev)
        for i, x in enumerate(ids):
            E[i, : len(x)] = self.emb(torch.tensor(x, device=self.dev))  # right padding (after the real tokens: harmless)
        pos = torch.arange(self.pos, self.pos + Lmax, device=self.dev).view(1, 1, -1).expand(3, N, -1)
        with self.fast_attn():
            out = self.lm(inputs_embeds=E, position_ids=pos, past_key_values=cache, use_cache=True,
                          output_hidden_states=hidden_layers is not None)
        h = out.last_hidden_state[torch.arange(N), last]
        logits = self.m.lm_head(h)
        res = dict(logits=logits.float(), cache=cache, last=last, Lmax=Lmax)
        if hidden_layers is not None:
            res["hidden"] = {L: out.hidden_states[L][torch.arange(N), last].float() for L in hidden_layers}
        return res

    @torch.no_grad()
    def generate(self, question, max_new=16, row=0):
        """greedy decode of one question from the snapshot (generative baseline; batch 1 so the GDN state never sees pads)"""
        r = self.ask([question], rows=[row]); cache = r["cache"]
        p = self.pos + r["Lmax"]; nxt = r["logits"].argmax(-1); out = []
        eos = {self.tok.convert_tokens_to_ids(IM_END), self.tok.eos_token_id}
        for _ in range(max_new):
            if int(nxt) in eos:
                break
            out.append(int(nxt))
            o = self.lm(inputs_embeds=self.emb(nxt[:, None]), position_ids=torch.full((3, 1, 1), p, device=self.dev),
                        past_key_values=cache, use_cache=True)
            nxt = self.m.lm_head(o.last_hidden_state[:, -1]).argmax(-1); p += 1
        return self.tok.decode(out, skip_special_tokens=True).strip()


def yes_no_ids(tok):
    y = [tok.encode(s, add_special_tokens=False)[0] for s in ("Yes", "yes", " Yes")]
    n = [tok.encode(s, add_special_tokens=False)[0] for s in ("No", "no", " No")]
    return sorted(set(y)), sorted(set(n))


def p_yes(logits, yid, nid):
    """p(yes) restricted to the yes/no tokens"""
    ly = torch.logsumexp(logits[:, yid], -1); ln = torch.logsumexp(logits[:, nid], -1)
    return torch.sigmoid(ly - ln)
