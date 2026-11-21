# video08: Qwen3.5-0.8B streaming video decision experiments (P2 / P3)

Frozen Qwen3.5-0.8B run as a stream: one forward per frame group (64 visual tokens, sliding 2-frame tubelet), bounded
full-attention KV (sink + last 4 s at full rate + 4–60 s at 2 fps), GDN state carried through the whole video, question
blocks forked from the current state and never written back. P2 measures the zero-training baselines (M6–M8, the
deployment protocol, linear probes); P3 trains only a readout (latent tokens + heads, optionally question-block LoRA).
Pre-registered success / stop rules: [`PREREG.md`](PREREG.md).

| File | What |
|---|---|
| `stream.py` | streaming engine; `tests/test_stream.py` checks it against the HF processor and the native forward |
| `vio.py`, `feed.py` | decoding on the exact k/fps grid, tubelet pairs, write loop |
| `evals/m6.py` `m7.py` `m8.py` | fps 2 vs 30 zero-shot; memory span with inserted word cards; 60-min stability |
| `evals/deploy.py`, `report.py` | stream × N rules protocol (FA/h, FPR@recall, latency, ECE, N-question cost) |
| `probe.py` | linear probes on question-free features |
| `heads.py`, `tasks.py`, `train_s1.py`, `evals/p3_eval.py` | S1 readout, training in the inference shape, heads vs generative |
| `jobs.sh`, `q.sh` | submission to the shared A100 queue (`~/glance-queue/glance-video.tsv`) |

## Data and licences

Everything is used for internal research only. Evaluation and training sets are split by video id; the Charades ids used
by Video-Index sources (`meta/decon/exclusions.json` of `Vision-Decision/glance-video-v0`) are removed from training.
No Video-Index video or question is used for training, tuning, or sent to any external service.

| Data | Source (download) | Licence / terms | Use |
|---|---|---|---|
| Charades v1 480p + annotations | AI2 S3 (`Charades_v1_480.zip`; hf-mirror copy `lmms-eval/charades_sta` was throttled) | AI2 "License for Non-Commercial Use": academic / non-profit research, no redistribution | train split (minus VI ids): S1 training, probe training; test split: evaluation only |
| Charades-STA | TALL annotations (via `glance-video-v0/meta/ann`) | follows Charades (non-commercial) | temporal queries; test split eval only |
| TimeLens-Bench annotations | hf-mirror `TencentARC/TimeLens-Bench` (json only) | see dataset card (research) | reference re-annotations (not yet used) |
| RepCount-A (LLSP) | hf-mirror `lmms-lab-eval/repcounta-lance` | no licence stated; academic use only | train (minus pHash-contaminated clips): count head; test: eval only |
| PushupBench | hf-mirror `anonymousatom/pushupbench` | see dataset card (research) | evaluation only |
| Qwen3.5-0.8B | hf-mirror `Qwen/Qwen3.5-0.8B` | Apache-2.0 | frozen backbone |
