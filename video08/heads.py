"""S1 readout: learnable latent tokens appended to each question block + small heads; optional LoRA that is active ONLY in
the question-block forward (the video path always runs the original weights, so the stream state is the frozen model's).

Per question row:  [question tokens][latent_0 .. latent_{K-1}][right padding]
  latent 0            presence / refusal  (sigmoid)
  latents 1..S        temporal slots: objectness + start / end bin distributions over [0, t_now] (B bins), Hungarian
  latent S+1          count over 0..C-1 (classification)
  latent S+2          "which one": query vector dotted with the hidden state at the end of each option
Hidden states are read at `layers` (concatenated); padding sits after the latents so it never reaches them.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LORA = {"on": False}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=32):
        super().__init__()
        self.base = base; self.r = r; self.s = alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))

    def forward(self, x):
        y = self.base(x)
        if LORA["on"]:
            y = y + (x.to(self.A.dtype) @ self.A.t() @ self.B.t()).to(y.dtype) * self.s
        return y


def lora_modules(lm):
    return [m for m in lm.modules() if isinstance(m, LoRALinear)]


def set_lora(mods, params):
    """point every LoRALinear at another adapter (params = [A0, B0, A1, B1, ...] in add_lora order): a free swap"""
    for i, m in enumerate(mods):
        m.A, m.B = params[2 * i], params[2 * i + 1]


def sft_loss(st, model, tok, items, rows):
    """generative SFT on the question block: rows = [question ids][answer ids, <|im_end|>]; CE on the answer tokens.
    The forked stream state is read-only (no gradient into the video path)."""
    from video08.tasks import answer_text
    end = tok.convert_tokens_to_ids("<|im_end|>")
    seqs = [(x["ids"], tok.encode(answer_text(x), add_special_tokens=False) + [end]) for x in items]
    N = len(seqs); L = max(len(q) + len(y) for q, y in seqs); dev = st.dev
    E = torch.zeros(N, L, st.emb.weight.shape[1], device=dev, dtype=st.dtype)
    for i, (q, y) in enumerate(seqs):
        E[i, : len(q) + len(y)] = st.emb(torch.tensor(q + y, device=dev))
    cache = st.fork(N, rows)
    pos = torch.arange(st.pos, st.pos + L, device=dev).view(1, 1, -1).expand(3, N, -1)
    h = st.lm(inputs_embeds=E, position_ids=pos, past_key_values=cache, use_cache=True).last_hidden_state
    bi, ti, tg = [], [], []
    for i, (q, y) in enumerate(seqs):
        for k, t in enumerate(y):
            bi.append(i); ti.append(len(q) - 1 + k); tg.append(t)
    logits = model.lm_head(h[torch.tensor(bi, device=dev), torch.tensor(ti, device=dev)]).float()
    return F.cross_entropy(logits, torch.tensor(tg, device=dev))


def add_lora(lm, r=16, alpha=32, targets=("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                                          "gate_proj", "up_proj", "down_proj")):
    params = []
    for name, mod in list(lm.named_modules()):
        for t in targets:
            child = getattr(mod, t, None)
            if isinstance(child, nn.Linear):
                w = LoRALinear(child, r, alpha); setattr(mod, t, w); params += [w.A, w.B]
    return params


def mlp(i, o, h=1024):
    return nn.Sequential(nn.LayerNorm(i), nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class Readout(nn.Module):
    def __init__(self, emb_weight, D=1024, layers=(12, 24), slots=4, bins=32, counts=64):
        super().__init__()
        self.layers, self.S, self.B, self.C = tuple(layers), slots, bins, counts
        self.K = 1 + slots + 2
        mu, sd = emb_weight.float().mean(0), emb_weight.float().std(0)
        self.latent = nn.Parameter((mu + sd * torch.randn(self.K, D, device=emb_weight.device)).to(emb_weight.device))
        Din = D * len(self.layers)
        self.presence = mlp(Din, 1)
        self.slot = mlp(Din, 1 + 2 * bins)
        self.count = mlp(Din, counts)
        self.choice_q = mlp(Din, 512); self.choice_k = mlp(Din, 512)

    def forward(self, st, qids, opt_ends=None, rows=None):
        stream_rows = rows  # which batched stream each question reads (None: stream 0)
        """st: Stream (snapshot = current state). qids: list of token-id lists. opt_ends: per row list of token indices
        (inside the question) whose hidden states are the options for the choice head. Returns dict of head outputs."""
        N = len(qids); dev = st.dev; K = self.K
        lens = [len(x) for x in qids]; Lmax = max(lens) + K
        E = torch.zeros(N, Lmax, self.latent.shape[1], device=dev, dtype=st.dtype)
        for i, x in enumerate(qids):
            E[i, : lens[i]] = st.emb(torch.tensor(x, device=dev))
        E = E.clone()
        lat = self.latent.to(st.dtype)
        rows = torch.arange(N, device=dev)
        idx = torch.tensor(lens, device=dev)[:, None] + torch.arange(K, device=dev)[None]
        E = E.index_put((rows[:, None].expand(-1, K), idx), lat[None].expand(N, -1, -1))
        cache = st.fork(N, stream_rows)
        pos = torch.arange(st.pos, st.pos + Lmax, device=dev).view(1, 1, -1).expand(3, N, -1)
        out = st.lm(inputs_embeds=E, position_ids=pos, past_key_values=cache, use_cache=True, output_hidden_states=True)
        hs = out.hidden_states
        H = torch.cat([hs[L][rows[:, None], idx].float() for L in self.layers], -1)  # (N, K, Din)
        res = dict(presence=self.presence(H[:, 0]).squeeze(-1))
        sl = self.slot(H[:, 1:1 + self.S])  # (N, S, 1+2B)
        res["obj"], res["start"], res["end"] = sl[..., 0], sl[..., 1:1 + self.B], sl[..., 1 + self.B:]
        res["count"] = self.count(H[:, 1 + self.S])
        if opt_ends is not None and any(opt_ends):
            q = self.choice_q(H[:, 2 + self.S])
            res["choice"] = []
            for i, ends in enumerate(opt_ends):
                if ends:
                    k = self.choice_k(torch.cat([hs[L][i, ends].float() for L in self.layers], -1))
                    res["choice"].append(k @ q[i] / math.sqrt(q.shape[-1]))
                else:
                    res["choice"].append(None)
        return res


# ------------------------------------------------------------------ losses / decoding
def soft_bins(x, B, sigma=1.0):
    """target distribution over B bins for a normalised position x in [0,1] (Gaussian, sigma in bins)"""
    c = torch.arange(B, device=x.device).float() + 0.5
    d = torch.exp(-0.5 * ((c[None] - x[:, None] * B) / sigma) ** 2)
    return d / d.sum(-1, keepdim=True)


def slot_loss(obj, start, end, segs, B):
    """obj (S,), start/end (S,B) logits; segs: list of (s,e) normalised. Hungarian match on -log p; returns loss"""
    from scipy.optimize import linear_sum_assignment
    S = obj.shape[0]
    if not segs:
        return F.binary_cross_entropy_with_logits(obj, torch.zeros_like(obj))
    segs = segs[:S]
    g = torch.tensor(segs, device=obj.device).float().clamp(0, 1 - 1e-4)
    ts, te = soft_bins(g[:, 0], B), soft_bins(g[:, 1], B)
    ls, le = F.log_softmax(start, -1), F.log_softmax(end, -1)
    cost = -(ls @ ts.t()) - (le @ te.t()) - F.logsigmoid(obj)[:, None]  # (S, G)
    r, c = linear_sum_assignment(cost.detach().cpu().numpy())
    tgt = torch.zeros_like(obj); tgt[torch.as_tensor(r, device=obj.device)] = 1
    l_obj = F.binary_cross_entropy_with_logits(obj, tgt)
    r = torch.as_tensor(r, device=obj.device); c = torch.as_tensor(c, device=obj.device)
    l_b = -(ls[r] * ts[c]).sum(-1).mean() - (le[r] * te[c]).sum(-1).mean()
    return l_obj + 0.5 * l_b


def decode_slots(obj, start, end, B):
    """-> list over slots of (p_obj, s, e) normalised (expected value of the bin distribution)"""
    c = (torch.arange(B, device=obj.device).float() + 0.5) / B
    s = (F.softmax(start, -1) * c).sum(-1); e = (F.softmax(end, -1) * c).sum(-1)
    return [(float(torch.sigmoid(o)), float(min(a, b)), float(max(a, b))) for o, a, b in zip(obj, s, e)]
