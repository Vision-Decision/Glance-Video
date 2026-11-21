"""Exact attention without a boolean mask (SDPA with a bool mask falls back to a ~7x slower kernel on the G4 / RTX PRO
6000, found by the latency prototype in glance-engine video-latency-p1). The new queries attend (a) every cached key
before them - unmasked fused kernel that also returns the log-sum-exp - and (b) the new keys causally - a small explicit
softmax; the two parts are merged by log-sum-exp. Registered as attn_implementation="split"."""
import torch
import torch.nn.functional as F

_BACKEND = {}


def _flash(q, k, v, s):
    r = torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, False, False, scale=s); return r[0], r[1]


def _eff(q, k, v, s):
    r = torch.ops.aten._scaled_dot_product_efficient_attention(q, k, v, None, True, 0.0, False, scale=s)
    return r[0], r[1][..., : q.shape[2]]


def window_attention(q, k, v, s):
    if "fn" not in _BACKEND:
        ref = F.scaled_dot_product_attention(q, k, v, scale=s).float()
        for name, fn in (("flash", _flash), ("efficient", _eff)):
            try:
                o, l = fn(q, k, v, s)
                if l.shape == q.shape[:3] and (o.float() - ref).abs().max() < 2e-2:
                    _BACKEND["fn"] = fn; print("split attention window backend:", name, flush=True); break
            except Exception:
                continue
    o, l = _BACKEND["fn"](q, k, v, s)
    return o, l.float()


def split_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    if not getattr(module, "is_causal", True) or "cu_seqlens" in kw:  # the vision tower: bidirectional, packed
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        return sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kw)
    B, H, q, d = query.shape; KH = key.shape[1]; G = H // KH; n = key.shape[2]
    s = scaling if scaling is not None else d ** -0.5
    if q == 1 or n == q:  # decode step (sees everything) or no cache (pure causal)
        k_, v_ = key.repeat_interleave(G, 1), value.repeat_interleave(G, 1)
        o = F.scaled_dot_product_attention(query, k_, v_, scale=s, is_causal=(n == q and q > 1))
        return o.transpose(1, 2).contiguous(), None
    qf = query.reshape(B, KH, G * q, d)
    o1, l1 = window_attention(qf, key[:, :, : n - q], value[:, :, : n - q], s)
    kl, vl = key[:, :, n - q:], value[:, :, n - q:]
    s2 = torch.matmul(qf.float(), kl.float().transpose(-1, -2)) * s  # (B, KH, G*q, q)
    i = torch.arange(q, device=query.device).repeat(G)
    s2 = s2.masked_fill(torch.arange(q, device=query.device)[None] > i[:, None], float("-inf"))
    l2 = torch.logsumexp(s2, -1)
    o2 = torch.matmul(torch.exp(s2 - l2[..., None]), vl.float())
    l = torch.logaddexp(l1, l2)
    o = o1.float() * torch.exp(l1 - l)[..., None] + o2 * torch.exp(l2 - l)[..., None]
    return o.to(query.dtype).reshape(B, H, q, d).transpose(1, 2).contiguous(), None


def register():
    from transformers import AttentionInterface
    AttentionInterface.register("split", split_attention)
