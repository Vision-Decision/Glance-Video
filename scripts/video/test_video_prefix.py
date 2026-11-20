"""Glance-Video sanity check with real weights (GPU).

1. Token layout: our video prefix ids (dv/data/video.py) == the HF Qwen3.5 processor's ids for a chat message holding
   the same frames as one video (apply_chat_template + processor(videos=...)), up to the end of the video.
2. Numerics: shared-prefix forward (causal mask, no LoRA) + TypedHead at init reproduces HF's own next-token logits
   on the label tokens for (video prefix + noul / choice block), with HF's own video rope index.

python scripts/video/test_video_prefix.py [--model_id Qwen/Qwen3.5-4B] [--frames 8] [--dt 0.5]
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch

from dv.config import Config
from dv.data import video as V
from dv.data.mixed import MixedDataset, collate_mixed
from dv.model.build import build
from dv.model.v1forward import run_model_v1
from train import to_dev

ap = argparse.ArgumentParser()
ap.add_argument("--model_id", default="Qwen/Qwen3.5-4B")
ap.add_argument("--frames", type=int, default=8)
ap.add_argument("--dt", type=float, default=0.5)
ap.add_argument("--max_vis", type=int, default=96)
a = ap.parse_args()

# synthetic clip: a red square moving right over a gradient, 256x320, 8 fps
T0 = 32
H, W = 256, 320
fr = np.zeros((T0, H, W, 3), np.uint8)
fr[..., 2] = np.linspace(0, 200, W, dtype=np.uint8)[None, None]
for t in range(T0):
    x = 10 + 8 * t
    fr[t, 100:160, x:x + 60, 0] = 255
V._CACHE["/fake/clip.mp4"] = fr
idx = list(range(0, 2 * a.frames, 2))

cfg = Config().update(dict(task="v1", model_id=a.model_id, attn_mask="causal", lora_r=0, grad_ckpt=False,
                           video_max_vis=a.max_vis))
net, heads, proc = build(cfg, "cuda", lora=False)
net.eval(); heads.eval()
recs = [dict(src="t", video="clip.mp4", frames=idx, dt=a.dt, max_vis=a.max_vis,
             questions=[dict(type="noul", instr="Does the red square move to the right in this clip?", target=1.0),
                        dict(type="choice", instr="Which way does the red square move?",
                             options=["left", "right", "up", "down", "it does not move"], target=[0, 1, 0, 0, 0])])]
ds = MixedDataset(None, "/fake", proc, cfg, train=False, recs=recs)
tok = ds.tok
items = [ds[0]]
it = items[0]

# ---- 1. token layout vs processor
g = it["video_grid_thw"]
n_per = int(g[1] * g[2] // 4)
ours_pre = []
for kind, ids in it["segs"]:
    ours_pre += ids if kind == "ids" else [tok.pad_vid] * (ids[0] * ids[1])
frames = V.gather_frames(recs[0], "/fake")
w, h = V.resize_dims(frames.shape[2], frames.shape[1], a.max_vis, 2.0)
x = torch.nn.functional.interpolate(torch.from_numpy(frames).permute(0, 3, 1, 2).float(), size=(h, w), mode="bicubic",
                                    align_corners=False).clamp(0, 255).round().to(torch.uint8)
msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "Q"}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
fps = 1.0 / a.dt
try:
    from transformers.video_utils import VideoMetadata
    md = VideoMetadata(total_num_frames=len(frames), fps=fps, frames_indices=list(range(len(frames))))
    pin = proc(text=[text], videos=[x], video_metadata=[md], do_resize=False, do_sample_frames=False, return_tensors="pt")
except Exception as e:  # older API
    print("metadata API failed:", repr(e)[:200])
    pin = proc(text=[text], videos=[x], do_resize=False, do_sample_frames=False, fps=[fps], return_tensors="pt")
ref_ids = pin["input_ids"][0].tolist()
qpos = len(ref_ids) - 1 - ref_ids[::-1].index(tok.ve)          # last vision_end = end of the video
ref_pre = ref_ids[:qpos + 1]
same = ref_pre == ours_pre
print("processor grid", pin.get("video_grid_thw"), "ours", g.tolist())
print("prefix ids equal:", same, len(ref_pre), len(ours_pre))
if not same:
    print("REF :", tok.tok.decode(ref_pre[:60]), "...", tok.tok.decode([i for i in ref_pre if i != tok.pad_vid])[:400])
    print("OURS:", tok.tok.decode(ours_pre[:60]), "...", tok.tok.decode([i for i in ours_pre if i != tok.pad_vid])[:400])
pv_diff = (pin["pixel_values_videos"].float() - it["pixel_values_videos"].float()).abs().max().item() \
    if pin["pixel_values_videos"].shape == it["pixel_values_videos"].shape else float("nan")
print("pixel_values_videos max diff:", pv_diff)

# ---- 2. numerics vs HF forward
batch = to_dev(collate_mixed(items, tok), "cuda")
batch.pop("meta", None)
with torch.no_grad():
    res = run_model_v1(net, heads, batch, train=False)
ours = res["typed_logits"].float()
groups = tok.label_token_groups()
maxdiff = 0.0
for k, (q, labs) in enumerate(zip(it["q_ids"], it["labs"])):
    ids = torch.tensor([ours_pre + q], device="cuda")
    mm = torch.zeros_like(ids)
    mm[ids == tok.pad_vid] = 2
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = net.hf(input_ids=ids, attention_mask=torch.ones_like(ids),
                     pixel_values_videos=it["pixel_values_videos"].cuda().to(torch.bfloat16),
                     video_grid_thw=g[None].cuda(), mm_token_type_ids=mm)
    lg = out.logits[0, -1].float()
    ref = torch.stack([torch.logsumexp(lg[torch.tensor(groups[l], device="cuda")], 0) for l in labs])
    mine = ours[k, :len(labs)]
    d = ((ref - ref.logsumexp(0)) - (mine - mine.logsumexp(0))).abs().max().item()
    maxdiff = max(maxdiff, d)
    print(f"q{k} labels={len(labs)} max|dlogp|={d:.4f} ours={[round(v, 3) for v in mine.softmax(0).tolist()]} "
          f"hf={[round(v, 3) for v in ref.softmax(0).tolist()]} top_tok={tok.tok.decode([int(lg.argmax())])!r}")
print("MAXDIFF", maxdiff, "OK" if (maxdiff < 0.15 and same) else "MISMATCH")
