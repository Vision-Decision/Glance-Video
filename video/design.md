# Video-Index + `vdm`: experiment design (draft, 2026-11-18)

**Claim to test.** Giving a fixed agent (same planner LLM, same harness, same budget) one extra tool, `vdm`,
raises Video-Index accuracy and/or cuts time and cost. `vdm` is our single-prefill decision model plus ffmpeg:
it scans short clips with yes/no questions to get P(t) curves (`find`, `count`), checks claims (`verify`), locates
boxes (`locate`) and estimates camera motion (`ego-motion`). There is no dedicated temporal-interval head.
The failure analysis (`analysis/failure_analysis.md`) gives the upper bound on what perception help can add.

## 1. Single-variable comparison

| Config | Planner | Harness | Tools | Notes |
|---|---|---|---|---|
| **base** | P | H | official toolset (shell, ffmpeg/ffprobe, Python + OpenCV/Pillow/NumPy, faster-whisper, image viewer) | our local replica (`sandbox/`) |
| **+vdm** | P | H | base + `vdm` (a CLI on PATH plus a ~15-line paragraph in the system prompt) | nothing else changes |
| **vdm-choose-only** | none | none | `vdm` answers the MCQ by itself: for each option, a yes/no scan over the clips, then choose = argmax over options of the aggregated P | lower bound, no LLM; shows what the tool alone knows |
| blind | P | H | question + options only, no video | language prior |

- The only difference between base and +vdm is the tool and its description. Same image tag, same harness version,
  same system prompt otherwise, same timeout (3600 s v1.1; 1800 s for v1.0 as in the paper), same concurrency and
  CPU/RAM per container. The `vdm` GPU time is counted inside the agent's wall clock.
- Planners: (1) an open model through Ollama (DeepSeek V4.1 Flash, which we already run in the sandbox) as the
  cheap main planner; (2) one frontier planner (Claude Opus 5.5 in Claude Code) **only after the user approves the
  spend**. Report the two separately; the claim is "+vdm helps planner P", not a cross-planner comparison.
- **Runs:** ≥3 independent runs per (config, planner). Seeds are not controllable through the CLIs, so runs are
  repeated sessions. Report the mean and the per-run spread.

## 2. Data

- **Main set: v1.0 (840 items)**, its normalized videos, as an agent task (Harbor `video` protocol: video in the
  container, shell tools, no network, 1 attempt, 1800 s). 840 items give a paired 95% CI of roughly ±3–4 pp on the
  difference; v1.1 alone gives roughly ±8 pp, too wide to be the headline.
- **Hard set: v1.1 (144 items)**, original videos with audio, 3600 s, exactly the published protocol, so numbers sit
  next to the published table of 21 agents.
- Also report v1.1 by capability group and by source benchmark family (UrbanVideo-Bench, S-EMBER, ALLVB, egocentric
  counting), where the failure analysis expects the gain.

## 3. Statistics

- **Paired bootstrap** over items (10,000 resamples) for Δacc = acc(+vdm) − acc(base). With R runs per config, the
  item score is the mean over runs (0, 1/R, …, 1), so the bootstrap carries both item and run variance.
  Also report a sign-flip permutation p-value and McNemar counts (items fixed vs broken) on the majority over runs.
- Stratified variant: resample within capability groups (keeps the group mix).
- Pre-register: primary metric = v1.0 accuracy with the open planner; one primary comparison (base vs +vdm);
  everything else is secondary and labelled exploratory.
- Minimum detectable effect: v1.0 around 3–4 pp; v1.1 around 8–10 pp at 80% power, so v1.1 alone is not
  conclusive unless the effect is large.

## 4. Controls

1. **Option-permutation control.** In v1.1, 59/144 answers are the last option and 29/71 five-option answers are E.
   Run each config also on a permuted copy (letters reassigned by `random.Random("42|<item_id>")`, the
   Harbor/blind convention; schema letters unchanged). Report accuracy on original and permuted order and the
   letter distribution of predictions. A gain that disappears under permutation is a position effect, not perception.
   Report the position-only heuristic score (failure analysis) as a reference line.
2. **Blind score.** Question and options only, no video, mean over four permutations (official `blind` protocol),
   for each planner. Report Δ(base − blind) and Δ(+vdm − blind) to show how much each config uses the video.
3. **Decontamination (76 source benchmarks).** `vdm`'s model must not have seen Video-Index items or videos:
   - Build a list of the 76 source benchmarks (v1.0 `benchmark` field and the repo's `configs/adapters`) and exclude
     their train **and** test splits from all `vdm` training data, matched by dataset name, by video SHA-256
     (`video_id` / `video_v1_0`), by YouTube id (`source_video`), and by perceptual hashes of frames sampled at 1 fps
     (pHash, Hamming ≤ 6) against our training video and image pools.
   - Text: 13-gram overlap of questions/options with our yes/no training prompts.
   - The Video-Index videos are licensed for evaluation only and are never used for training or calibration.
     Temperature scaling and thresholds for P(t) are fitted on held-out non-Video-Index data.
   - Publish the exclusion list and the hash-match counts.
4. **Tool-description control.** Run base with a dummy `vdm` that returns "unavailable" to check that the extra
   prompt text alone does not change behaviour.

## 5. Ablations (each on v1.0 + v1.1, open planner, ≥2 runs)

- Remove one `vdm` subcommand at a time: `find`, `count`, `verify`, `locate`, `ego-motion`.
- **Calibrated probabilities vs hard labels:** `vdm` returns P(yes) curves and calibrated intervals vs only yes/no
  labels at threshold 0.5. Tests whether calibration is what the planner uses.
- Clip length and stride of the scan (e.g. 2 s/1 s vs 4 s/2 s vs 8 s/4 s) — accuracy vs latency.
- Raw Qwen3.5-4B yes/no logprobs (the prototype) vs our trained decision model, with the same wrapper.
- Optional: `vdm` with a cost cap (max seconds of video scanned per call).

## 6. Metrics reported per config

Accuracy (overall, per group, per source), no-answer count split into timeout / error / no structured answer,
wall time per question (mean, median, p90; timeouts counted as the limit), tokens (input, cached, output) and cost
per question (list price per provider, with the cached-token lower/upper bound like the official table), number of
tool calls and `vdm` calls, `vdm` GPU seconds, letter distribution of predictions, blind accuracy, permuted accuracy.
All traces kept in the official trace format (`stdout.jsonl`, `result.json`, `command.json`, `prompt.txt`).

## 7. Order of work

1. Base runs of the open planner on v1.1 (done once, see `sandbox/README.md`), then v1.0 (needs the 840 videos).
2. `vdm` prototype with raw Qwen3.5-4B logprobs; check on timeout/counting items offline.
3. +vdm runs ×3, controls, ablations. 4. Frontier planner only with explicit approval.

## Questions for the maintainer

1. Would you accept a **custom-tool track** (same planner + harness, with/without an extra tool) next to the
   current leaderboard? What disclosure do you want (tool code, model weights, training data list)?
2. Is the v1.0 agentic protocol (Harbor, 1800 s, normalized videos) still the reference, or should agent results
   use the original videos as in v1.1? Will there be original videos for all 840?
3. Can you share the exact runner (bwrap/landlock policy, harness configs) so our replica matches? The traces show
   bwrap without `--unshare-net` for Claude/Gemini: how was "no web access" enforced for the shell?
4. The working directory path contains the item id (e.g. `.../calls/ALLVB_1040`), visible to the agent. Is that
   intended? It leaks the source benchmark name.
5. How were concurrency and machine size set (CPU cores per call; Whisper on CPU)? Timeouts on long movies depend
   on it; we would like to report time on comparable hardware.
6. Position bias: 59/144 v1.1 answers are the last option. Would you publish a permuted v1.1 or accept
   permuted-order results as a robustness column?
7. Are repeated runs planned (run-to-run variance), and would you report CIs on the table?
8. Decontamination: is the list of the 76 source benchmarks with their split names available as a file?
9. For a tool that runs a GPU model: is GPU time allowed inside the 3600 s, and on what hardware should it run?
10. Scoring edge cases found in the traces: one timed-out run that had already written the correct letter
    [Video-Index item details redacted] (ALLVB_1040)
    [Video-Index item details redacted]
[Video-Index item details redacted] (V2P-Bench_988)
    [Video-Index item details redacted] (ALLVB_14547)
    another option, with frame evidence).
12. The machine running the official calls was heavily loaded (median load average ≈103 in the Opus 5.5 traces;
    Whisper large-v3 on CPU in 49/144 runs). How many of the timeouts would you attribute to infrastructure?
13. Would you take a PR with an open-model baseline (DeepSeek through Ollama in Claude Code), and in which
    form (traces + scores row with a "non-official harness backend" note)?
