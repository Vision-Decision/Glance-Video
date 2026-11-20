# Native-video base model on Video-Index choose-only (2026-11-18)

Source: Glance-Video (vdm v1) pilot evaluation, run on Colab G4 (`gv-a`/`gv-b`); raw results in HF `Vision-Decision/glance-video-v0` under `results/`.
Model: Qwen3.5-4B, native video input (not single frames). Scoring as in SNAPSHOT.md §choose-only: argmax = option with the highest probability, argmin = lowest.

| Set | Input | argmax correct | argmin correct |
|---|---|---|---|
| v1.1 (144 items) | clips8 | 3/144 | 53–59% (range over input modes) |
| v1.1 (144 items) | global | 5/144 | 53–59% |
| v1.0 (random 200 items, chance 30%) | clips8 / global | 11% / 17.5% | 39–48% |
| v1.1, vdm v1 pilot (500 steps) | — | 9/144 | 62% |

Reading: the anti-correlation is not specific to single-frame scoring; a native-video VLM of the kind the v1.0 screen targets shows it too.
