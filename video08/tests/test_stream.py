"""Equivalence checks for the streaming engine (CPU fp32 or GPU).  python video08/tests/test_stream.py [cpu|cuda] [video.mp4]
 1. patchify == the HF Qwen3-VL video processor's pixel rows
 2. stream with keep_all, 1 group per forward == 6 groups per forward (GDN chunked == recurrent, KV identical)
 3. stream (ts every group, non-overlapping pairs) == native model forward on the processor's input
 4. key rotation by -s == RoPE at position p - s
"""
import sys, os
if (sys.argv[1] if len(sys.argv) > 1 else "cpu") == "cpu":
    sys.modules["fla"] = None  # triton kernels need CUDA: force the torch GDN path
import torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from video08.stream import Stream, load, patchify, yes_no_ids, p_yes
from video08 import vio

dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
dt = torch.float32 if dev == "cpu" else torch.bfloat16
torch.manual_seed(0)
model, tok = load(device=dev, dtype=dt)
from transformers import AutoProcessor
proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B")

if len(sys.argv) > 2:
    fr, times = vio.decode(sys.argv[2], 2.0, 256, 0, 3.0)
else:
    fr = (torch.rand(6, 3, 256, 256) * 255).to(torch.uint8)
fr = fr[:6]
# 1. patchify
vp = proc.video_processor
o = vp(videos=[fr.numpy()], return_tensors="pt", do_resize=False, do_sample_frames=False,
       video_metadata=[dict(fps=2.0, total_num_frames=6, frames_indices=list(range(6)))])
pairs = fr.float().div(255).reshape(3, 2, 3, 256, 256)
px, grid = patchify(pairs)
print("1 patchify max|diff|", (px - o["pixel_values_videos"]).abs().max().item(), "grid", o["video_grid_thw"].tolist())

yid, nid = yes_no_ids(tok)
q = "Is there a person in the video? Answer yes or no."
# 2. per-group vs chunked
def run(groups_per_fwd, pairs_in, ts_every=0.5, times=None):
    s = Stream(model, tok, keep_all=True, ts_every=ts_every)
    vis, g = s.encode(pairs_in)
    times = times if times is not None else [0.5 * (i + 1) for i in range(len(pairs_in))]
    for i in range(0, len(times), groups_per_fwd):
        s.write(vis[i:i + groups_per_fwd], times[i:i + groups_per_fwd], g)
    return s, s.ask([q])["logits"]
P6 = vio.pairs(fr)
s1, l1 = run(1, P6); s6, l6 = run(6, P6)
print("2 per-group vs chunked: max|dlogit|", (l1 - l6).abs().max().item(), "p_yes", p_yes(l1, yid, nid).item(), p_yes(l6, yid, nid).item())

# 3. native: processor builds "<t seconds><vs><video_pad>*64<ve>" per temporal group
msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": q}]}]
text = proc.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
inp = proc(text=[text], videos=[fr.numpy()], return_tensors="pt", do_resize=False, do_sample_frames=False,
           video_metadata=[dict(fps=2.0, total_num_frames=6, frames_indices=list(range(6)))]).to(dev)
print("native text:", repr(proc.tokenizer.decode(inp["input_ids"][0]).replace("<|video_pad|>", ""))[:400])
with torch.no_grad():
    ln = model(**inp).logits[0, -1].float()
# stream with the same layout: timestamps every group, non-overlapping pairs, times = processor's timestamps
ts_native = [0.25, 1.25, 2.25]
s, ls = run(1, pairs.to(dt) if False else pairs, ts_every=0.0, times=ts_native)
print("3 stream vs native: max|dlogit|", (ls[0] - ln).abs().max().item(), "argmax", ls[0].argmax().item(), ln.argmax().item(),
      "p_yes", p_yes(ls, yid, nid).item(), p_yes(ln[None], yid, nid).item())

# 4. rotation
rot = model.model.language_model.rotary_emb
x = torch.randn(1, 1, 5, 256, device=dev)
def rope(x, p):
    pos = torch.full((3, 1, x.shape[2]), p, device=dev)
    c, sn = rot(x, pos)
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    return apply_rotary_pos_emb(x, x, c.float(), sn.float())[1]
s.cache.layers[s.full_layers[0]].keys = rope(x, 1000)  # pretend cache
saved = s.full_layers; s.full_layers = [saved[0]]
s._rotate_keys(700, torch.ones(5, dtype=torch.bool, device=dev))
print("4 rotate(-700) vs rope(300): max|diff|", (s.cache.layers[saved[0]].keys - rope(x, 300)).abs().max().item())
