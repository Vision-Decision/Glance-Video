# Video-Index replica: results snapshot (2026-11-18 09:05 UTC)

Partial results. Every run walks the same seeded random order (random.Random(0) over sorted item ids, on the 144 v1.1 items or on the 64-item hard set), so a partial run is a random subset of its set. Scoring: correct = answered with the key; a timeout or a missing answer is wrong (README rule). Opus items stopped by the subscription rate limit and +vdm items hit by the vdm server outage are **not scored** and are re-queued. Hard set = `sandbox/hard60.json` (64 items, perception-bound / timeout-heavy for the official agents), so accuracies on it are far below the full-set numbers.

## 1. Accuracy per condition

| Condition | Runs | Scored n | Correct | Acc. | of which hard set | Running | Timeouts | Errors | Min/item mean / median |
|---|---|---|---|---|---|---|---|---|---|
| DeepSeek V4.1 Flash, no tool | pilot10, full1 | 66 | 11 | 16.7% | 4/31 | 12 | 21 | 1 | 40.8 / 40.2 |
| DeepSeek V4.1 Flash, +vdm | ds_vdm_hard | 8 | 0 | 0.0% | 0/8 | 4 | 6 | 0 | 55.0 / 60.0 |
| Gemini 4 Hydrogen, no tool | gem_pilot5, gem_hard25 | 25 | 11 | 44.0% | 11/25 | 0 | 0 | 5 | 10.6 / 8.6 |
| Gemini 4 Hydrogen, +vdm | gem_vdm_pilot5, gem_vdm_hard25 | 24 | 7 | 29.2% | 7/24 | 1 | 0 | 4 | 12.5 / 11.9 |
| Claude Opus 5.5 (subscription), no tool | opus_pilot5, opus_hard10 | 9 | 4 | 44.4% | 4/9 | 1 | 0 | 0 | 23.4 / 19.5 |
| Claude Opus 5.5 (subscription), +vdm | opus_vdm_hard10 | 3 | 1 | 33.3% | 1/3 | 1 | 0 | 0 | 24.1 / 19.0 |

DeepSeek no-tool on the full v1.1 set so far: 11/66 (16.7%), of 144. For reference, the published table: Opus 5.5 47.9%, Gemini 4 Argon 41.0%, Haiku 5.5 27.8%, Haiku 4.5 14.6%; chance ≈ 24.8%.

By group (DeepSeek no-tool): perception 2/14, reasoning_knowledge 2/17, spatial_physical 4/17, temporal 3/18

## 2. Paired comparisons (common finished items only)

| B vs A | n common | Acc. A | Acc. B | Δ = B − A (pp) | 95% bootstrap CI | B fixes / B breaks |
|---|---|---|---|---|---|---|
| DeepSeek V4.1 Flash, +vdm vs DeepSeek V4.1 Flash, no tool | 3 | 1/3 | 0/3 | -33.3 | [-100.0, +0.0] | 0 / 1 |
| Gemini 4 Hydrogen, +vdm vs Gemini 4 Hydrogen, no tool | 24 | 11/24 | 7/24 | -16.7 | [-33.3, -4.2] | 0 / 4 |
| Claude Opus 5.5 (subscription), +vdm vs Claude Opus 5.5 (subscription), no tool | 3 | 2/3 | 1/3 | -33.3 | [-100.0, +0.0] | 0 / 1 |
| Gemini 4 Hydrogen, no tool vs DeepSeek V4.1 Flash, no tool | 10 | 2/10 | 3/10 | +10.0 | [-20.0, +40.0] | 2 / 1 |
| Claude Opus 5.5 (subscription), no tool vs DeepSeek V4.1 Flash, no tool | 3 | 1/3 | 2/3 | +33.3 | [-100.0, +100.0] | 2 / 1 |
| Claude Opus 5.5 (subscription), no tool vs Gemini 4 Hydrogen, no tool | 8 | 5/8 | 4/8 | -12.5 | [-50.0, +25.0] | 1 / 2 |
| Claude Opus 5.5 (subscription), no tool vs official claude-opus-5-5 (published trace) | 9 | 3/9 | 4/9 | +11.1 | [+0.0, +33.3] | 1 / 0 |
| Gemini 4 Hydrogen, no tool vs official gemini-4-argon (published trace) | 25 | 8/25 | 11/25 | +12.0 | [-12.0, +36.0] | 7 / 4 |
| Gemini 4 Hydrogen, no tool vs official gemini-3.8-flash (published trace) | 25 | 10/25 | 11/25 | +4.0 | [-24.0, +32.0] | 7 / 6 |
| DeepSeek V4.1 Flash, no tool vs official claude-haiku-4-5 (published trace) | 66 | 7/66 | 11/66 | +6.1 | [-4.5, +16.7] | 8 / 4 |
| DeepSeek V4.1 Flash, no tool vs official claude-haiku-5-5 (published trace) | 66 | 18/66 | 11/66 | -10.6 | [-21.2, +0.0] | 4 / 11 |

With n this small, a CI that spans 0 means no detectable effect; one fix or break moves Δ by 100/n pp. Official rows are a replication check: same items, same prompts, published traces (official rule credits a letter given at timeout; ours does not, which matters for at most a few items).

## 3. Cost and usage

- **Gemini API (billed, list price, from gateway token counts): $21.76 total** — gem_hard25 $7.10, gem_pilot5 $2.62, gem_smoke $0.08, gem_vdm_hard25 $8.39, gem_vdm_pilot5 $3.56, gemtest $0.00, gemtest2 $0.02. Budget ≈ $100, of which ≈ $65 reserved for vdm v1.
- DeepSeek (Ollama cloud flat subscription; cost-equivalent at DeepSeek list price): $10.44 — ds_vdm_hard $2.36, full1 $6.27, pilot10 $1.80.
- Claude Opus 5.5 (subscription, not billed): Claude-Code-reported API-equivalent $80.69 for scored items; latest rate-limit event: 5-hour window 21%, 7-day window 4% (cap for this study: ~60%).

## 4. vdm usage

- Gateway-level vdm HTTP requests (all +vdm runs and health checks): 492; status {200: 418, 599: 74}. `/score` latency mean 6.7 s, median 6.6 s, p90 10.7 s (n=417).
- Failed vdm requests (599 = no server answered) at UTC 06:05, 06:09, 06:10, 06:11, 06:33, 06:34, 06:35, 06:36, 07:11, 08:08, 08:09, 08:10, 08:11, 08:21, 08:22, 08:23, 08:24, 08:25, 08:26, 08:39, 08:40, 08:41.
- DeepSeek V4.1 Flash, +vdm: vdm sub-commands per scored item mean 1.8, median 0.0, items with no vdm call 5/8.
- Gemini 4 Hydrogen, +vdm: vdm sub-commands per scored item mean 2.0, median 2.0, items with no vdm call 8/24.
- Claude Opus 5.5 (subscription), +vdm: vdm sub-commands per scored item mean 1.0, median 1, items with no vdm call 1/3.
- Compromised by the vdm outage and re-queued (not scored): 4 — ds_vdm_hard_deepseek-v4.1-flash_UrbanVideo-Bench_5348, gem_vdm_hard25_gemini-4-hydrogen_S-EMBER_2431, gem_vdm_hard25_gemini-4-hydrogen_S-EMBER_7308, gem_vdm_hard25_gemini-4-hydrogen_UrbanVideo-Bench_5348.

## 5. vdm v0 choose-only (no LLM): v1.1 vs v1.0

`vdm choose`, 32 uniform frames, each frame answers the MCQ alone, mean option probability; **argmin** = pick the option with the lowest mean probability. Chance = mean of 1/#options.

**v1.0 is not a clean control.** Video-Index v1.0 was itself built adversarially against frame-based VLMs (repo `docs/pool.md`, `docs/compose.md`): the pool screen removes every item that an 8B attacker answers correctly with margin > log 2 from a **single frame** (frame 16 of 32), or a 2B attacker from 32 frames or 32 shuffled frames, or from text only. The 840 released items are then the *hardest first*, ranked by the worst-case attacker margin percentile (median 0.14, max 0.23). Margin = log p(key) − mean log p(other options), so the released items are those where the frame models put *low* probability on the key. A single-frame scorer like Glance should therefore score below chance on v1.0 too. v1.1 (a subset that agents fail) inherits the effect. A clean control needs unscreened items from the source benchmarks.

| Set | Group | n | choose (argmax) | argmin | chance |
|---|---|---|---|---|---|
| v1.1 | all | 144 | 3 (2.1%) | 96 (66.7%) | 24.8% |
| v1.1 | perception | 33 | 1 (3.0%) | 28 (84.8%) | 32.7% |
| v1.1 | reasoning_knowledge | 37 | 1 (2.7%) | 17 (45.9%) | 20.9% |
| v1.1 | spatial_physical | 39 | 0 (0.0%) | 23 (59.0%) | 22.7% |
| v1.1 | temporal | 35 | 1 (2.9%) | 28 (80.0%) | 23.8% |
| v1.0 (partial 155/840) | all | 155 | 5 (3.2%) | 107 (69.0%) | 30.0% |
| v1.0 | perception | 44 | 2 (4.5%) | 30 (68.2%) | 31.6% |
| v1.0 | reasoning_knowledge | 37 | 0 (0.0%) | 24 (64.9%) | 23.7% |
| v1.0 | spatial_physical | 27 | 2 (7.4%) | 17 (63.0%) | 25.8% |
| v1.0 | temporal | 47 | 1 (2.1%) | 36 (76.6%) | 35.9% |

**Choose-only bug audit (independent, 07:50–08:10).** Synthetic clip (big colored rectangle, key at every position, 2–7 options, same docker + v0 client path): argmax correct 63/63, mean p(key) 0.999. Server↔Glance.ask: same prompt builder (`typed_block`), same label groups 0..n−1 = A.., same softmax/temperature; server selftest compares the batched path with Glance.ask. Keys: test.jsonl `answer` = LETTERS[answer_idx] for 144/144 and = official `expected` for 144/144; the choose-only harness keeps the item's option order (no shuffling).
Permutation test on 10 v1.1 items (options reversed, probabilities mapped back to the original letters): the argmax stays on the same option *text* in 7/10; the key stays in the lower half of the option probabilities in 9/10. The low score follows the content of the key, not its letter or position.

