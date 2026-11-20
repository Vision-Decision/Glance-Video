# glance-video

**A 4B video model that answers typed questions about short clips, served to video agents as a search tool.**
Glance-Video scores every 2-second clip of a video for one visual phrase in a single forward pass; `vdmscan` hands
the agent the most likely moments with a calibrated verdict, so the agent can check a few moments instead of
extracting frames across the whole video.
This repository trains the model, serves the tool, and replicates the agent protocol of Video-Index, a long-video
question-answering benchmark for agents, to test whether the tool helps.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/video-scan-loop-dark.svg">
  <img width="100%" alt="Agent loop: a video agent (any LLM planner) calls vdmscan find video.mp4 with a short visual phrase. The scan server (Glance-Video, 4B) has split the video into 2-second clips (up to 600), each encoded once and cached, and one forward pass asks the phrase of every cached clip, giving a per-clip probability (drawn schematically). The tool returns the top-k moments with probabilities, a verdict (PRESENT calibrated to about 10% false alarms on videos without the event) and one contact sheet with 4 frames per moment. The agent looks only at those moments, then asks the next question or answers. Tool contract: video/vdm/gv/TOOL_scan.md." src="assets/video-scan-loop-light.svg">
</picture>

<sub>The agent turns its question into short visual events and asks the tool where they happen; the clip states are
cached, so each new phrase costs one forward pass over the whole video. Contract:
[`video/vdm/gv/TOOL_scan.md`](video/vdm/gv/TOOL_scan.md).</sub>

<table>
<tr>
<td align="center" width="33%"><h2>92.9%</h2>repetition-count accuracy on Countix<br><sub>base Qwen3.5-4B: 29.4%, chance 21.4%</sub></td>
<td align="center" width="33%"><h2>0.192</h2>relative counting error on RepCount-A<br><sub>base Qwen3.5-4B: 0.524 (lower is better)</sub></td>
<td align="center" width="33%"><h2>9.0%</h2>false alarms of the PRESENT verdict on held-out Charades scans<br><sub>detection 76.7%; a fixed 0.5 threshold: 32.1% false alarms, 94.9% detection</sub></td>
</tr>
</table>

The model and the tool's verdicts are measured above; whether the tool raises a Video-Index agent's accuracy is not
shown yet ([Key results](#key-results)).

## Quickstart

```bash
git clone https://github.com/Vision-Decision/glance-video.git && cd glance-video
pip install -e .        # pulls glance-core and glance-engine v0.2.0 from GitHub
bash tests/cpu.sh       # tiny random model on CPU: scan engine + vdmscan HTTP server end to end, then a compile check
```

`tests/cpu.sh` ends with `SCAN_EQUIVALENCE_OK` and `compile check: all files OK`. Nothing in it needs a GPU or a
checkpoint.

Serving the trained model to an agent (GPU VM). **Organisation members only for now:** the Glance-Video checkpoint is
not public yet, so the download below needs Vision-Decision access on the Hub.

```bash
hf download Vision-Decision/glance-train-ckpts glance-video-v0/ckpt_final.pt --local-dir /content/ckpt
python video/vdm/gv/scan_server.py --ckpt /content/ckpt/glance-video-v0/ckpt_final.pt --port 8004
VDMSCAN_URL=http://localhost:8004 video/vdm/gv/vdmscan --help   # agent-facing CLI; planner contract: video/vdm/gv/TOOL_scan.md
```

## What is here, and why

Our reading of the official Video-Index agent traces is that long-video agents often fail by not finding the moment that matters, rather than by misreading it once found. A model that
can answer "is this happening in this clip?" for every clip of a video in one forward pass, with a calibrated
probability, is a natural search tool for such an agent. This repo asks two questions:

1. **Can a 4B decision model learn typed video questions?** Glance-Video is initialised from Glance v1.1 and trained
   on clips with counting, ordering, camera-motion and moment questions, after removing anything that overlaps
   Video-Index.
2. **Does the scan tool make an agent better on Video-Index?** The harness runs the official prompts and limits with
   any planner, with and without `vdmscan`, on the same items, and compares them pairwise.

## Key results

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/video-calibration-dark.svg">
  <img width="100%" alt="False-alarm and detection rates of the set-level 'is the event anywhere in this scan' verdict on held-out Charades-STA test videos, by clips per scan (natural length median 29, then 32, 64, 128, 256, 512, 900 clips). A fixed P greater than 0.5 threshold: false alarms 32% at natural length, 39% at 32 clips, 60% at 64, 85% at 128, 96% at 256, 100% at 512 and 900. Calibrated PRESENT verdict: 9.0%, 9.6%, 9.1%, 9.2%, 10.0%, 10.4%, and 15% at 900 clips (n=127). Detection rate of the calibrated verdict: 77% at natural length falling to 39% at 900 clips. Source: video/gate/results/gate_stats.json." src="assets/video-calibration-light.svg">
</picture>

<sub><b>The tool's “present” answer keeps its false-alarm rate as scans get longer.</b> Long scans here are short
Charades-STA test videos concatenated, with the threshold fitted for each length on separate videos. A fixed
probability threshold almost always fires on long scans; the calibrated verdict holds its 10% target up to 512 clips
(15% at 900, beyond the tool's 600-clip cap), at the price of lower detection on very long scans. Script:
[`assets/video_calib.py`](assets/video_calib.py).</sub>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/video-results-dark.svg">
  <img width="100%" alt="Left: accuracy on held-out typed video questions, Glance-Video vs zero-shot Qwen3.5-4B, with chance level: repetition count on Countix val 92.9 vs 29.4 (chance 21.4, n=1,232); which event happens, Charades, 68.6 vs 55.1 (chance 25, n=312); camera motion, Kinetics-700, 79.6 vs 70.4 (chance 19.6, n=270); event order, Charades, 83.6 vs 75.3 (chance 50, n=73); arrow of time, Kinetics-700, 56.0 vs 50.7 (chance 50, n=797). Right: RepCount-A relative MAE 0.192 vs 0.524 (n=152). Sources: outputs/gv0/full_typed.json, base_typed.json, full_repcount.json, base_repcount.json." src="assets/video-results-light.svg">
</picture>

<sub><b>Training on typed clip questions beats the base model on all five tasks shown, most of all on counting.</b> Both models
read the same native video prefix; the base answers zero-shot through its own label tokens (no published reference
models in this chart). Arrow of time stays close to
chance. Negative-only and yes/no variants of these tasks are in the same files. Result files: [`outputs/gv0/`](outputs/gv0) (copied from the Hub run folder `results/gv0/`).</sub>


| What | Result | Source |
|---|---|---|
| Countix repetition count, 5-way choice accuracy (chance 0.21): Glance-Video vs base Qwen3.5-4B | **0.929** vs 0.294 (n = 1,232) | [`outputs/gv0/full_typed.json`](outputs/gv0/full_typed.json) (`raw.count`), `base_typed.json` |
| RepCount-A, relative MAE (\|pred − gt\| / gt, lower is better), per-window argmax: Glance-Video vs base | **0.192** vs 0.524 (n = 152) | [`outputs/gv0/full_repcount.json`](outputs/gv0/full_repcount.json), `base_repcount.json` (`pred_argmax.MAE`) |
| Calibrated PRESENT verdict on held-out Charades-STA scans (α = 0.1): false alarms / detection | 9.0% / 76.7% (fixed 0.5 threshold: 32.1% / 94.9%); clip AUROC 0.899 | [`video/gate/results/gate_stats.json`](video/gate/results/gate_stats.json) (`ever.natural`, `clip_auroc`) |
| Scan engine vs one-shot scoring on Charades-STA test videos (6 videos, 182 clips, 501 clip–query pairs) | 99.0% same decisions, mean \|ΔP\| 0.005 | [`video/vdm/gv/README.md`](video/vdm/gv/README.md) |
| Video-Index replica, Gemini 4 Hydrogen planner with vs without `vdm` v0 (24 paired hard-set items) | **−16.7 pp**, 95% CI [−33.3, −4.2] | [`video/results/SNAPSHOT.md`](video/results/SNAPSHOT.md) |
| Video-Index replica v2, DeepSeek V4.1 Flash planner with vs without `vdmscan` (64 paired hard items) | 6/64 → 9/64, **+4.7 pp**, 95% CI [−1.6, +12.5]; 4 fixes, 1 break | [`video/results/vi_scan_v2/summary.md`](video/results/vi_scan_v2/summary.md) §4 |
| Same, Opus 5.5 planner | 31/64 → 31/64, +0.0 pp, 95% CI [−9.4, +9.4]; 5 fixes, 5 breaks | same file, §5 |

In the Video-Index agent replica the tool does not yet raise accuracy: on 64 paired hard items (ones the official frontier
agents failed; DeepSeek's base accuracy there is below chance), with a fixed harness,
a DeepSeek agent went from 6 to 9 correct (+4.7 pp, 95% CI [−1.6, +12.5]) and an Opus 5.5 agent stayed at 31; the
earlier frame-question tool `vdm` v0 made a Gemini agent worse on 24 paired items and is frozen
([`video/results/vi_scan_v2/summary.md`](video/results/vi_scan_v2/summary.md)).

## Reproduce

| Step | Entry point | Data / output |
|---|---|---|
| Clip pool and decontamination against Video-Index (dataset, id, SHA-256, 1-fps pHash) | `scripts/video/prep_clips.py`, `decon_list.py`, `gen_records.py`, `push_shards.py` (on gen-host) | `Vision-Decision/glance-video-v0` (`shards/`, `records_full/`, `meta/decon/`) |
| Training (from v1.1) | `MODE=full bash scripts/video/vm_train.sh` → `train.py --cfg configs/video_v0.json` from glance-core | `Vision-Decision/glance-train-ckpts:glance-video-v0/` |
| Offline evals | `python scripts/video/eval_video.py typed\|sta\|repcount\|tb\|vi --out X.json` | `Vision-Decision/glance-train-ckpts:results/gv0/` |
| Video-Index offline gate (calibration on non-Video-Index data) | `python video/gate/vi_gate.py`, `video/gate/vi_gate_e2e.py` | — |
| Agent sandbox (Docker per question, official flags) | `video/sandbox/README.md`; `python3 video/sandbox/run.py --run NAME [--vdmscan]`; v2 paired runs: `video/sandbox/v2/run_v2.sh NAME --set hard60` (both arms at once) | `video/results/` |
| Paired statistics | `python3 video/sandbox/paired.py --a RUN_A --b RUN_B --ref full1` | `video/results/` |

**Video-Index data is evaluation-only.** Its videos and frames are never used for training, calibration or
thresholds, and never copied to this repo or to Hugging Face; `video/sandbox/fetch.sh` downloads them on the host
that runs the agents. Traces stay on that host because they contain frames. See `video/design.md`.

## Layout and status

| Path | Contents | Status |
|---|---|---|
| `scripts/video/`, `configs/video_v0.json` | Glance-Video data, training and evals | v0 done |
| `video/vdm/gv/` | `scan_server.py` (scan engine over HTTP), `vdmscan` CLI, `core_gv.py` one-shot backend | v2 in use by the paired run |
| `video/vdm/` (rest) | `vdm` v0 server and tool (frozen), training-free v1 prototype, `choose` baselines | frozen |
| `video/sandbox/` | Video-Index agent replica: gateway, runner, paired analysis, v2 scripts | active |
| `video/gate/` | Video-Index offline gate and Glance-Video pilot comparison (moved from the monorepo's `scan/`) | done |
| `video/results/` | Snapshot, deviations from the official protocol, v2 paired results | v2 in progress |
| `outputs/blog/` | Blog figures and widget data (Big Buck Bunny frames, COCO scenes) | done |

Per-item Video-Index outputs (answers, item text, agent traces), the item fixtures of the vdm offline tests and the
meeting deck are not in git, because Video-Index items may not be redistributed; they are kept in a private dataset
of the organisation.

## Related repositories

| Repository | Contents |
|---|---|
| [glance-core](https://github.com/Vision-Decision/glance-core) | The Glance model: training, data, evaluation |
| [glance-engine](https://github.com/Vision-Decision/glance-engine) | Inference engine and the scan engine `vdmscan` is built on |
| [Glance](https://github.com/Vision-Decision/Glance) | Public model release and inference package |

## License

Code: Apache-2.0 ([LICENSE](LICENSE)). Model weights follow their model cards. Video-Index is used under its
evaluation-only terms and none of its data is in this repository.
