> **Moved (2026-11-22):** development continues in `Vision-Decision/glance-stream` (package `glance_stream`); this directory is kept as history only.

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

## 排期（2026-11-22 更新）

| 时间 | 内容 | 机器 |
|---|---|---|
| 进行中 | S1a / S1b（1600 步 × 4 段）→ P3 评测（fps2 / fps30） | a100 队列 |
| 进行中 | M7 跑满 60 次，M8 用修好的窗口重跑，严格逐帧延迟与 N 问吞吐重测 | Colab G4 |
| 进行中 | M6 剩余臂、部署协议、探针特征；R1 零样本网格 | a100 队列（lane E1–E3） |
| 进行中 | B0 数据：HC-STVG v2、MeViS、LaSOT 子集、RefCOCO 标注 | a100 CPU |
| 约 10:00 后 | B1 框头（从头训 vs 图像 v1.4_D_0.8B 初始化），随后 B2、B3、B4 | a100 队列 |
| S1 出结果后 | P4 设计（`P4_DESIGN.md`：写入门控 vs 加长 KV）和 R2 混合训练，等确认后开训 | — |

### S1 白天续训和 P3 的硬件约定

- 07:40 A100 让出时，队列给 S1a / S1b 发停止信号，两者存 `ckpt_last.pt` 后退出（exit 3）。07:41 起 Mac 上的 `handoff.sh` 把两个检查点（每个约 140 MB）拷到 Colab G4，`g4_day.sh` 用同样的参数从检查点续训到 1200 步，并在 A100 队列里把 S1 和 P3 评测标记为已转走，今晚不再重复跑。G4 上的 Charades train 视频和 RepCount-A 由 `g4_prep.sh` 预先从官方 S3 / HF 拉好。
- **P3 的配对比较只在 G4 上做**：`evals/p3_eval.py` 在同一个进程、同一个流状态、同一组 token 上同时算头读出和生成式答案（含各自耗时），所以两边自然是同一硬件。A100 上不再跑 P3 评测。
- VidSTG 只做评测（报告里的 m_vIoU），不进训练、不调参。
