# vdm v1: Glance-Video backend

`vdm` v0 (`video/vdm/vdm`, `server.py`) stays frozen. `video/vdm/v1/` is the training-free clip prototype
(Glance-4B-preview on a multi-image prefix). This directory serves the **trained** clip model, Glance-Video
(`configs/video_v0.json`; data, training and evals in `scripts/video/`; checkpoint
`Vision-Decision/glance-train-ckpts:glance-video-v0/ckpt_final.pt`).

- `core_gv.py`: `GVScorer`, which scores a clip as a native Qwen3.5 video prefix with the typed head. It implements
  `ClipBackend`:
  - `prepare(video_id, clips, t0s, dt)` decodes the clips and runs the vision tower once.
  - `scan(video_id, questions)` returns per-clip P(yes) or option probabilities for every question.
  - `drop(video_id)` frees the cache.

  `score` / `score_multi` are the one-shot path.
- `server_gv.py`: `/score_clips` (the `vdm1` client API, unchanged), plus `/prepare`, `/scan`, `/drop`.
  `--backend module:Class` swaps in any other `ClipBackend`.
- `scan_server.py`: the scan engine (`scan/engine.py`). Each clip's prefix state is cached and one forward pass
  answers a question for all clips. It adds the `curve` / `when` / `count` / `ever` operations with set-level
  guarantees.

```bash
python video/vdm/gv/server_gv.py --ckpt /content/ckpt/glance-video-v0/ckpt_final.pt --port 8002
python video/vdm/gv/server_gv.py --base --port 8003      # zero-shot Qwen3.5-4B on the same input (baseline)
python video/vdm/gv/scan_server.py --ckpt /content/ckpt/glance-video-v0/ckpt_final.pt --port 8004
```

The scan engine agrees with `score_multi` (`scripts/video/test_scan_vs_core.py`, 6 Charades-STA test videos,
182 clips, 501 compared clip–query pairs): mean |ΔP| 0.005, max 0.046, 99.0% of decisions the same, logit correlation 0.9997.
`core_gv`'s cached `scan` equals its one-shot path exactly.
