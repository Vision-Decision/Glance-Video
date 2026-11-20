# Video-Index agent + vdmscan, v2: a paired comparison that measures the tool, not the infrastructure

Status (2026-11-20 01:30 UTC, complete): diagnosis and fixes done; DeepSeek pilot passed (8 items x 2 arms); **DeepSeek 64-item
paired run done (§4)**; Opus 5.5 64-item paired run done (§5). Code: Vision-Decision/
glance-video (video/sandbox, video/vdm/gv). Data: per-item results in `pilot/`, `hard64/`, `opus_hard64/` (no traces: they contain
video frames); HF Vision-Decision/vi-scan-v2.

## 1. Why the v1 paired run (ds_base_hard vs ds_scan_hard, 2026-11-18) is not evidence about the tool

v1 setup: DeepSeek V4.1 Flash (Ollama cloud) inside Claude Code 2.1.300 headless, official VI v1.1 prompt and flags,
64 hard items, 4 + 4 concurrent containers with 2 CPUs each, 60-min limit, timeout or no answer = wrong.

| where the time went (items with a result) | base | scan |
|---|---|---|
| items finished / timed out | 22 / 22 | 9 / 22 |
| timeouts overlapping an Ollama quota-exhausted window | 20 / 22 | 21 / 22 |
| planner calls that exhausted the gateway's 14 retries (each blocked ~59 min, holding an in-flight slot) | 62 | 63 |
| finished items: median wall | 23.9 min | 27.8 min |
| finished items: median planner calls (≈8 s each, effort max) | 102 | 79 |
| finished items: share of wall in planner latency / in tool execution | 73% / 23% | 47% / 25% |
| finished items: accuracy | 3/22 | 1/9 |
| vdmscan first call (encode) median / max; cached call median | - | 19 s / 174 s; 0.5 s |
| scan-server bank evictions during the run | - | 76 |

Root causes, by impact:
1. **Planner quota, not agents.** Ollama served for ~1.5-2 h, then returned 429 for ~2 h; the account is now at its
   weekly limit. The gateway retried each 429 for ~59 min while holding an in-flight slot, so one blocked call became
   a 60-min timeout (41/44 timeouts).
2. **No step budget.** ~100 planner calls per finished item = 15-20 min of pure planner latency.
3. **CPU work in the 2-CPU sandbox.** Full-movie whisper (30+ min for a 100-min film), 24-fps frame dumps, `sleep`
   polling; in the scan arm also whole-video decoding for the first scan (keyframe-only decode for long videos).
4. **Scan output not decision-ready.** On whole movies `when` answered "never (calibrated NO)" whenever the 600-clip
   max-statistic threshold was not cleared, with no candidates; agents fell back to manual frame hunting.
5. **Bank memory.** ~56 MB per clip, ~33 GB per 600-clip bank; the 90 GB server budget held 2-3 videos and finished
   runs' banks evicted in-flight ones.

Not causes: docker start-up (~1 s), the tunnel (no outage in the run), scan latency (26 min encode + 12 min scans
summed over the whole run). Per-item v1 numbers: `diag_v1_per_item.json`.

## 2. Fixes (v2 harness, both arms identical except the tool)

| problem | fix |
|---|---|
| quota stalls scored as wrong | planner = DeepSeek official API (`deepseek-flash` = V4.1 Flash, 2,500 concurrency; median call 2.9 s); quota/balance errors fail fast + fatal flag, backoff never holds a slot, per-item stall accounting; infra failures are requeued, never scored |
| unbounded agents | gateway budget: 40 tool steps (remaining steps announced for the last 8; at the cap only StructuredOutput may be called) and 25 min wall (remaining minutes announced after 12.5 min, answer forced at 22.5 min); hard limit 30 min |
| CPU work inside the clock | per video, once: whisper-small transcript mounted read-only as `transcript.tsv` (both arms); whole-video frames pre-extracted; scan arm's clip index built before the agent's clock starts (median 31 s, reported separately) |
| scan output | `vdmscan find`: k most likely moments with temporal NMS, calibrated p, a PRESENT / NOT DETECTED verdict, and ONE contact sheet of those moments in time order (joint ordered re-read); `when` lists best sub-threshold candidates instead of "never" |
| bank memory | banks tagged by owner and released when the item ends; server budget 150 GB (70 GB GPU) |
| arm drift | both arms run at the same time over the same seeded order (`--arms base,scan`) |

Deviations from the official VI protocol (apply to both arms): 30-min limit instead of 60, step/time budget,
cached transcript, DeepSeek official API instead of Ollama.

## 3. Pilot (8 hard items stratified by source × 2 arms, DeepSeek V4.1 Flash, effort max)

16/16 answered, 0 timeouts, 0 errors; median wall 11.7 min (base) / 11.2 min (scan); forced answers 1 + 1.
Accuracy base 1/8, scan 2/8 (2 fixes, 1 break; n = 8, not informative). 7/8 scan agents used the tool (find×21,
count×4, when×1); a scan call takes ~8 s incl. the contact sheet. Cost ≈ $0.05-0.06 per run (mostly peak price).
Planner calls include non-step requests, so they can exceed 40.

| item | key | base pred | scan pred | base min | scan min | base calls | scan calls | base out tok | scan out tok | scan ops |
|---|---|---|---|---|---|---|---|---|---|---|
| ALLVB_12953 | [Video-Index item details redacted] |
| ALLVB_20839 | [Video-Index item details redacted] |
| S-EMBER_3231 | [Video-Index item details redacted] |
| S-EMBER_346 | [Video-Index item details redacted] |
| UrbanVideo-Bench_5348 | [Video-Index item details redacted] |
| UrbanVideo-Bench_962 | [Video-Index item details redacted] |
| ViMUL-Bench_4920 | [Video-Index item details redacted] |
| VideoOdyssey_2669 | [Video-Index item details redacted] |
[Video-Index item details redacted] (S-EMBER_346)
the crossing (~140 s) correctly with `find`, then counted with `vdmscan count` in sub-windows; `count` merges
occurrences < ~2 s apart and gave 5 instead of 8. Dense-event counting should use frames, not `count`.

## 4. Paired results, 64 hard items, DeepSeek V4.1 Flash (primary evidence)

Run 2026-11-19 10:00-13:05 UTC (off-peak), 3 + 3 concurrent, same seeded order, 30-min limit, 40 steps / 25 min budget.

| | base (no tool) | + vdmscan |
|---|---|---|
| accuracy | 6/64 (9.4%) | 9/64 (14.1%) |
| paired Δ (scan − base), 95% bootstrap CI | | **+4.7 pp [-1.6, +12.5]**; fixes 4, breaks 1; McNemar p = 0.375 |
| answered / timeouts / errors | 64 / 0 / 0 | 64 / 0 / 0 |
| wall time median / mean | 6.8 / 8.0 min | 8.0 / 8.2 min |
| planner requests median; forced answers (cap) | 38; 7 | 38; 4 |
| input tokens (incl. cached) / output tokens per item | 1.13M / 34k | 1.07M / 38k |
| planner cost (DeepSeek list price, by call time) | $2.01 ($0.031/item) | $2.15 ($0.034/item) |
| frame-extraction steps per item (mean) | 40.1 | 32.2 |
| items using the tool; calls | - | 62/64; {'find': 313, 'when': 27, 'count': 24, 'reps': 4} |
| index build before the clock (median); scan call latency (median) | - | 10 s; 3.7 s |

By slice (n, base correct, scan correct, Δ pp [CI]):

| slice | n | base | scan | Δ pp |
|---|---|---|---|---|
| counting | 15 | 1 | 1 | +0.0 [+0, +0] |
| temporal | 8 | 1 | 3 | +25.0 [+0, +62] |
| perception | 9 | 0 | 0 | +0.0 [+0, +0] |
| spatial | 24 | 2 | 3 | +4.2 [+0, +12] |
| reasoning | 8 | 2 | 2 | +0.0 [-38, +38] |

Reading: the direction is positive (+4.7 pp, 4 fixes vs 1 break) but the CI includes 0; with base accuracy at 9% on this
hard set (items selected because frontier agents fail them, below the 20-25% chance level of 4-5 options), 64 pairs
cannot resolve an effect of this size. The tool replaced manual frame hunting (−20% frame-extraction steps) at no
time or cost penalty (+1.2 min median, +7% cost). Discordant items: fixes ALLVB_19237, ALLVB_3868, S-EMBER_3626,
S-EMBER_5054; break ALLVB_17054.

Per item (pred ✓ = correct; time; output tokens; cost):

| item | source | key | base | scan | base min | scan min | base out tok | scan out tok | base $ | scan $ | scan ops |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ALLVB_1040 | [Video-Index item details redacted] |
| ALLVB_10848 | [Video-Index item details redacted] |
| ALLVB_12052 | [Video-Index item details redacted] |
| ALLVB_12953 | [Video-Index item details redacted] |
| ALLVB_14547 | [Video-Index item details redacted] |
| ALLVB_14913 | [Video-Index item details redacted] |
| ALLVB_15618 | [Video-Index item details redacted] |
| ALLVB_17054 | [Video-Index item details redacted] |
| ALLVB_17443 | [Video-Index item details redacted] |
| ALLVB_19237 | [Video-Index item details redacted] |
| ALLVB_19787 | [Video-Index item details redacted] |
| ALLVB_20839 | [Video-Index item details redacted] |
| ALLVB_2121 | [Video-Index item details redacted] |
| ALLVB_23938 | [Video-Index item details redacted] |
| ALLVB_3868 | [Video-Index item details redacted] |
| ALLVB_429 | [Video-Index item details redacted] |
| ALLVB_5354 | [Video-Index item details redacted] |
| ALLVB_690 | [Video-Index item details redacted] |
| ALLVB_7117 | [Video-Index item details redacted] |
| ALLVB_9711 | [Video-Index item details redacted] |
| S-EMBER_1358 | [Video-Index item details redacted] |
| S-EMBER_1448 | [Video-Index item details redacted] |
| S-EMBER_1548 | [Video-Index item details redacted] |
| S-EMBER_2431 | [Video-Index item details redacted] |
| S-EMBER_2539 | [Video-Index item details redacted] |
| S-EMBER_26 | [Video-Index item details redacted] |
| S-EMBER_283 | [Video-Index item details redacted] |
| S-EMBER_3042 | [Video-Index item details redacted] |
| S-EMBER_3231 | [Video-Index item details redacted] |
| S-EMBER_346 | [Video-Index item details redacted] |
| S-EMBER_3626 | [Video-Index item details redacted] |
| S-EMBER_3638 | [Video-Index item details redacted] |
| S-EMBER_3823 | [Video-Index item details redacted] |
| S-EMBER_4991 | [Video-Index item details redacted] |
| S-EMBER_5054 | [Video-Index item details redacted] |
| S-EMBER_544 | [Video-Index item details redacted] |
| S-EMBER_5572 | [Video-Index item details redacted] |
| S-EMBER_6606 | [Video-Index item details redacted] |
| S-EMBER_6678 | [Video-Index item details redacted] |
| S-EMBER_7291 | [Video-Index item details redacted] |
| S-EMBER_7308 | [Video-Index item details redacted] |
| S-EMBER_7718 | [Video-Index item details redacted] |
| S-EMBER_7790 | [Video-Index item details redacted] |
| S-EMBER_8015 | [Video-Index item details redacted] |
| S-EMBER_8250 | [Video-Index item details redacted] |
| S-EMBER_826 | [Video-Index item details redacted] |
| S-EMBER_8655 | [Video-Index item details redacted] |
| S-EMBER_888 | [Video-Index item details redacted] |
| S-EMBER_9230 | [Video-Index item details redacted] |
| S-EMBER_9383 | [Video-Index item details redacted] |
| UrbanVideo-Bench_1010 | [Video-Index item details redacted] |
| UrbanVideo-Bench_1130 | [Video-Index item details redacted] |
| UrbanVideo-Bench_2432 | [Video-Index item details redacted] |
| UrbanVideo-Bench_3112 | [Video-Index item details redacted] |
| UrbanVideo-Bench_3940 | [Video-Index item details redacted] |
| UrbanVideo-Bench_4351 | [Video-Index item details redacted] |
| UrbanVideo-Bench_4657 | [Video-Index item details redacted] |
| UrbanVideo-Bench_4930 | [Video-Index item details redacted] |
| UrbanVideo-Bench_4973 | [Video-Index item details redacted] |
| UrbanVideo-Bench_53 | [Video-Index item details redacted] |
| UrbanVideo-Bench_5348 | [Video-Index item details redacted] |
| UrbanVideo-Bench_962 | [Video-Index item details redacted] |
| ViMUL-Bench_4920 | [Video-Index item details redacted] |
| VideoOdyssey_2669 | [Video-Index item details redacted] |

## 5. Opus 5.5, 64 hard items, paired (frontier-planner check)

Same harness and items, Claude Code 2.1.300 headless with `claude-opus-5-5`, effort max, 1 + 1 concurrent, run
2026-11-19 07:07 - 2026-11-20 00:46 UTC. The gateway cannot inject budgets into TLS traffic, so both arms get the wall
limit stated in the system prompt (30 min, start time) instead of the step/time budget. run.py waited out the 5-hour
subscription window above 85% and would have stopped above 70% of the 7-day window; the scan VM was shut down during
those waits and restored before each reset (restore 121-167 s).

| | base (no tool) | + vdmscan |
|---|---|---|
| accuracy | 31/64 (48.4%) | 31/64 (48.4%) |
| paired Δ, 95% bootstrap CI | | **+0.0 pp [−9.4, +9.4]**; fixes 5, breaks 5; McNemar p = 1.0 |
| answered / timeouts / errors | 64 / 0 / 0 | 64 / 0 / 0 |
| wall time median / mean | 6.4 / 7.4 min | 7.5 / 8.1 min |
| turns (median); tool calls (mean) | 45; 37.2 | 42; 37.0 |
| tokens per run (mean): cache-read / cache-write / output | 1.58M / 89k / 32k | 1.77M / 85k / 35k |
| frame-extraction steps (mean) | 30.6 | 29.1 |
| items using the tool; calls | - | 47/64; find 113, when 38, count 9 |

By source (base → scan correct): ALLVB 6 → 5, S-EMBER 19 → 18, UrbanVideo 5 → 7, VideoOdyssey 1 → 1, ViMUL 0 → 0.
Fixes: S-EMBER_2431, S-EMBER_26, UrbanVideo 3940 / 5348 / 962. Breaks: ALLVB_1040, S-EMBER_1358 / 826 / 9383,
UrbanVideo-Bench_3112. The stratified 8-item subset looked favourable early (1/8 → 4/8); the full 64 does not confirm it.

Subscription use: 128 runs moved the account's 7-day window 0.11 → 0.50 (≈0.3%/run, an upper bound since other agents
shared the account); the clean single pair measured ≈0.5%/run and ≈5% of the 5-hour window per run.

Reading across planners: with the weak planner (DeepSeek, 9% base) the tool points positive (+4.7 pp, 4 fixes / 1 break,
not significant); with the strong planner (Opus, 48% base) it is neutral (5 fixes / 5 breaks). Opus already finds the
moments by itself (fewer frame-extraction steps than DeepSeek even without the tool, 30.6 vs 40.1) and uses the tool
less (2.5 vs 5.8 calls per item). On these items the tool is at best a substitute for search the planner can already do.

## 6. Spend, caveats, next steps

- DeepSeek official API: pilot ≈ $0.9, hard-64 paired $4.17 (list price by call time; ≈ ¥36 together). The shared key
  was also drained by another project's 24-worker job (~¥25 in one hour); balance at 13:05 UTC: ¥2.81. A DeepSeek
  144-item paired run (~$9.4 off-peak) needs a top-up.
- Ollama cloud: the account is at its weekly usage limit (the root cause of the v1 failure).
- Calibration caveat (video/gate/results/gate_stats.json): `ever` / PRESENT is calibrated (false alarm 0.090 at
  α = 0.1 on natural negatives), but `when` is not FDR-controlled (mean FDP 0.30 vs q = 0.1 on held-out Charades);
  `count` merges events closer than ~2 s (an observed failure: S-EMBER_346 in the pilot).
- Protocol deviations from the official VI v1.1 run (both arms): 30-min limit, step/time budget (DeepSeek) or stated
  limit (Opus), cached whisper-small transcript, DeepSeek official API. Results are a paired tool comparison, not
  leaderboard numbers.
- Scan VM during subscription waits: `video/sandbox/v2/scan_vm_cycle.sh` stops the scan VM when the Opus run enters a
  5-hour-window wait longer than 45 min (once no agent is running) and restores it 30 min before the reset with
  `scan_vm.sh restore` (new G4, deps + checkpoint from HF, fresh localhost-only tunnel key, scan + whisper servers).
  Measured restore: 167 s to healthy through the tunnel; encode + scan verified on synthetic frames.
- Next (user decision, costs money): a DeepSeek top-up for the 144 set paired, which has more power and is less
  selected toward distractor-designed items.
