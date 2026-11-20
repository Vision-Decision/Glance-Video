# Protocol deviations and incidents (running log)

Replica of the Video-Index v1.1 agent protocol (see `video/sandbox/README.md`). This file lists every place where
our runs differ from the official protocol or from our own plan, and every incident that affected results.
Times are UTC, 2026-11-18.

## Standing deviations (by design)

1. **Models and harness access.** DeepSeek V4.1 Flash through Ollama cloud (Anthropic-compatible endpoint) under
   Claude Code 2.1.300; Gemini 4 Hydrogen through the Gemini API under Gemini CLI 0.64.0; Claude Opus 5.5 through
   the user's Claude subscription (OAuth) under Claude Code 2.1.300. The official runs used the Kaggle model proxy.
2. **Hardware/concurrency.** Each item gets 1.25 CPUs (pilot10 had 2 CPUs) on a shared 30-vCPU host. The host also
   runs unrelated jobs (GUI synthesis, k700 decoding, ffmpeg jobs on the v1.1 videos), and the load average was 25–75
   during the runs. Time per item is only roughly comparable across conditions, and slow CPU (Whisper large-v3)
   produces timeouts. The official host was also oversubscribed (load ~100), so neither side is clean.
3. **Scoring.** A timeout is wrong even when a letter was emitted (README rule). The official scores credit such
   letters (see `analysis/failure_analysis.md`). This changes at most a few items. Official comparisons in the
   snapshot use the official rule for the official traces.
4. **Egress.** Agent containers sit on an `--internal` Docker network. Their only reachable host is the per-condition
   gateway. The no-tool Gemini gateway (`vi-gateway-gem`) and the no-tool Opus gateway (`vi-gateway-opus`) also proxy
   `/vdm/*`, but no-tool containers do not have the `vdm` client or its description. From 07:15 the +vdm runs use their
   own gateways (`vi-gateway-vdm`, `vi-gateway-gemvdm`, `vi-gateway-opusvdm`), so each +vdm condition has its own
   network.
5. **Gemini CLI side calls.** Gemini CLI's loop-detection service calls `gemini-3-flash-preview`. The gateway allows
   only `gemini-4-hydrogen`, so these calls get a 403 (seen 2 times up to 07:10: gem_vdm_pilot5/S-EMBER_2539 and
   gem_hard25/ALLVB_14913). The main loop continues after the 403. The official Gemini runs presumably allowed it.
6. **+vdm adds one endpoint** (the vdm GPU server through gateway → relay → reverse SSH tunnel to a Colab VM). GPU
   time counts inside the 3600 s budget. vdm v0 (Glance-4B per-frame scorer) is frozen for all current +vdm runs.

## Incidents

### vdm outages (verified from the gateway log and the traces, 06:58–07:05)
- Gateway-level failures (599, no vdm server answered): **06:05:45–06:11:12** (5 requests) and
  **06:33:51–06:36:52** (17 requests). Out of 322 vdm requests up to 07:05, 22 failed. After 06:37 neither VM was
  serving: dv-vdm and dv-vdm2 had been reclaimed and recreated at 06:37–06:39. `runs/_compromised/outages.json`
  records these two windows. The coordinator's note (~06:20–06:40) roughly covers the second window.
- Each +vdm trace was scanned for vdm error output (`server error 50x`, `vdm server unreachable/unavailable`).
  **4 items saw vdm errors**, and the previous agent had already moved all 4 to `runs/_compromised/`. They are not
  scored and will be re-run: ds_vdm_hard/UrbanVideo-Bench_5348 (9 errors), gem_vdm_hard25/S-EMBER_2431 (2),
  gem_vdm_hard25/S-EMBER_7308 (4), gem_vdm_hard25/UrbanVideo-Bench_5348 (2).
- Items that ran during an outage window but saw no vdm error stay scored. ds_vdm_hard/S-EMBER_2539 and
  S-EMBER_3231 ran until 06:07 with no failed call. opus_vdm_hard10/S-EMBER_346's vdm calls were at 06:19–06:22.
  gem_vdm_hard25/UrbanVideo-Bench_962 never called vdm. gem_vdm_pilot5/S-EMBER_346 had no error in its trace.
- The 4 compromised items ran under the **old client** (3 tries, 3 s apart). At 06:39:53 the client in
  `vdmclient/vdm` was replaced by v0.1, which only adds retries: 502/503/504 and connection errors are retried with
  exponential backoff (2 s → 30 s) for up to 2 min. Scoring and model are unchanged. Items before and after 06:40 use
  different retry behaviour; this affects only robustness.
- All three +vdm runs (ds_vdm_hard, gem_vdm_hard25, opus_vdm_hard10) were found stopped at 06:51, with no live
  containers and no orphaned claims. They were resumed after the server came back (see below).

### Gateway source regression (found 07:08, fixed 07:12)
- When the previous agent added multi-server failover to `sandbox/gateway.py` (06:39), it deleted the `_google`
  method, the Gemini API pass-through. The running Gemini gateway still had the old file, because a bind-mounted file
  replaced on the host keeps the old inode inside the container, so no run was affected. Any gateway restarted
  in Gemini mode would have failed every call. The method was restored verbatim from the running container, merged
  with the failover code, and tested locally (failover to the second upstream, 503 + `retryable` when every upstream
  is down, 403 for a non-allowlisted Gemini model). It was then installed on gen-host and the laptop.

### Second vdm server not available (07:10)
- `dv-vdm2` (acct-7) was recreated at 06:37 and is now running the vdm **v1** agent's `eval_v1.py` (in `/content/v1`).
  It does not serve vdm v0 and we left it alone. Creating a replacement VM (`dv-vdm3`) was refused by Colab (the
  primary account is full). The pool rule says not to switch accounts to get around this. The +vdm runs therefore
  have **one** vdm v0 server (dv-vdm → :18080). The gateways list :18082 as a failover upstream, so a second server
  started later on :18082 is used without restarting anything.

### v1.0 choose-only
- First attempt (before 06:37) recorded 3 items as wrong (`pred: null`) because the server was unreachable. Those
  records were moved to `choose_only_v1_0.jsonl.errors.jsonl`. The script now leaves infrastructure failures
  unscored: it retries with backoff for up to ~15 min and then re-queues the item. It also walks the items in seeded
  random order, so partial results are a random subset. Settings are otherwise identical to the v1.1 choose-only run
  (32 frames, frozen v0 client `vdm/vdm`). v1.0 uses the **normalized** v1.0 videos; v1.1 used the original files.

### Opus subscription
- opus_pilot5 (before the plan upgrade): 4/5 items were rate-limited (five-hour window at 100%). They are not scored,
  and opus_pilot5 is not re-run (it is superseded by opus_hard10). After the upgrade (from 06:18): five-hour window
  ≤ 5%, seven-day window ≤ 1%.

### Gemini no-tool "errors"
- 3 gem_hard25 items (ALLVB_1040, ALLVB_14913, ALLVB_17054, all 1.5–2 h films) end with status `success` but no
  answer. Each time, the CLI ends right after the agent `read_file`s a long audio file (mp3/wav): the CLI calls
  `countTokens` and stops without another `generateContent`. This looks like the CLI's context-overflow guard. It is
  a harness behaviour (the same CLI 0.64.0 as the official Gemini runs) and not our infrastructure, so these items
  stay scored as wrong (no answer).

### vdm server redeploy (07:00–07:47)
- dv-vdm (acct-7, recreated 06:38) had Glance installed but no server, keep-alive or tunnel. The first redeploy
  loaded the model, but its long `exec` lost the connection and the kernel hung. A kernel restart then killed the
  server, because it had been started from the kernel's process group. `deploy_vm.py` was changed so that the
  server, keep-alive and tunnel run under `setsid`, and the script no longer blocks while the model loads. The
  server code is still the frozen v0 `server.py`, unchanged. Server back at 07:40 (selftest ok); reverse tunnel up
  at 07:47.
- The tunnel key in gen-host `~/.ssh/authorized_keys` belonged to the reclaimed VM. It was replaced by the new VM's
  key with the same restriction (`restrict,port-forwarding,permitlisten="127.0.0.1:18080",command="/bin/true"`).
  Backup: `authorized_keys.bak_0745`.
- The 599 at 07:11 in the gateway log is our own isolation test while no server was up; no agent was running.
- +vdm runs resumed at 07:47 (ds_vdm_hard 4×, gem_vdm_hard25 5×, opus_vdm_hard10 1×) on their own gateways
  (`vi-gateway-vdm`, `vi-gateway-gemvdm`, `vi-gateway-opusvdm`; each on its own `--internal` network; upstreams
  :18080, :18082). We checked from inside a container on each network: no DNS or internet, the vdm path works end to
  end, a non-allowlisted Gemini model gets a 403, and the Opus CONNECT proxy reaches only api.anthropic.com.
- v1.0 choose-only started at 07:48 with 3 parallel containers and was restarted at 07:58 with 5 (resumable; no
  scored item lost).

### Choose-only "bug" audit (08:00, requested by the coordinator)
- **Verdict: no bug in the code path. The anti-correlation is real and comes from how the benchmark was built.**
- Request: the exact prompt the model sees is
  `"\n\n{question}\n\nOptions:\nA. {opt0}\nB. {opt1}…\nAnswer with the letter of the best option."` + chat
  template, rendered by Glance's own `typed_block`. The server uses the same builder, the same label groups
  (0..n−1 = A..) and the same softmax/temperature as `Glance.ask`. The server's selftest checks the batched path
  against `Glance.ask` at every start ("selftest ok").
- Keys: in `test.jsonl`, `answer` = LETTERS[answer_idx] for 144/144 items and equals the official traces'
  `expected` for 144/144. The choose-only harness keeps the item's option order: no permutation, and A.. assigned in
  list order.
- Synthetic sanity, through the full docker + v0 client path (colored rectangle clip; key at every position;
  2/3/4/5/7 options; 3 colors): **63/63 correct**, mean p(key) 0.999.
- Permutation test on 10 random v1.1 items (options reversed): the argmax follows the option text in 7/10, and the
  key stays in the lower half of the probabilities in 9/10. The low probability follows the key's *content*, not its
  [Video-Index item details redacted] (ALLVB_17054)
[Video-Index item details redacted] (AoTBench_4027)
  [Video-Index item details redacted]
  visually right for the frame and wrong for the item.
- Why it is real: Video-Index **v1.0 itself** was screened against frame-based VLMs (`docs/pool.md`): every item
  that an 8B attacker answers with margin > log 2 from **one frame**, or a 2B attacker from 32 (shuffled) frames,
  was removed. The 840 kept items are the hardest by worst-case attacker margin (median percentile 0.14). They are
  items where frame models put low probability on the key. v1.1 is a further subset of these. On v1.0 so far (84/840),
  choose-only scores 1.2% (argmax) and 72.6% (argmin), so the effect is **not v1.1-specific**. v1.0 is therefore not a
  clean control. A clean test needs unscreened items from the source benchmarks.
- Minor findings, which do not explain the effect: (a) VidPair-Halluc items embed the option list inside the
  question, and their option strings are Python dict reprs (`{'label': 'A', 'text': …}`), so Glance sees the options
  twice and in an odd format (only a few items). (b) Typed blocks over 384 tokens are cut from the front (keeping
  [Video-Index item details redacted] (Video-MME-v2_1998)

### Gemini silent stop after audio read (open issue)
- 5/25 Gemini no-tool items (all ALLVB films) end with no answer right after `read_file` of an audio file of
  1.1–5 MB. 2 of the 7 items that read audio did not stop (files ≤ 1.1 MB). In the official Gemini 3.8 Flash traces
  (same CLI 0.64.0), 20 items read audio and **none** ended this way. `countTokens` from Hydrogen for these files is
  small (≈1–6k tokens), so the CLI's context-overflow guard is not the obvious cause. The cause is not established
  (Hydrogen + CLI interaction, or the gateway path for large inline requests). These items stay scored as wrong,
  per protocol, and the snapshot should be read with this in mind: Gemini no-tool would be 11/20 without them.

### Automation
- gen-host cron (`5 */3 * * *`, `sandbox/snapshot_cron.sh`) writes `~/videoindex/results/SNAPSHOT_<ts>.md` and
  `SNAPSHOT.md`, a report of +vdm items with vdm infrastructure errors (`flag_vdm_errors.py`, report only; move
  them by hand with `--move`, then relaunch the run), and a vdm health probe.

### vdmscan arm (Glance-Video scan tool), 2026-11-18/19
- **Arms.** `ds_base_hard` (no tool) and `ds_scan_hard` (+`vdmscan`), DeepSeek V4.1 Flash, hard set (64 items), same
  seeded order, `--concurrency 4 --cpus 2` per arm, **one shared gateway** (`vi-gateway-scan`, 5 upstream requests in
  flight) so both arms see the same Ollama load. The base arm started ~35 min before the scan arm (the end-to-end
  gate was still running); for most of the run both arms overlap. `full1` (no tool, earlier day) is kept as a
  run-to-run reference for the base arm.
- **Extra endpoint.** As for vdm v0: the gateway forwards `/vdm/*` to the scan server on one Colab G4
  (`dv-scan`) through a reverse ssh tunnel restricted to one listen port (gen-host 127.0.0.1:18084) and a relay on
  docker0 (172.17.0.1:18084). Frames are extracted in the agent container (2 CPUs, counted in its wall clock) and
  encoded on the GPU server; banks live in GPU/host RAM only and are never written to disk or used for training.
- **Calibration.** `CALIB.json` = Charades-STA test, the CAL half of the 700-video clip pilot of the served model
  (`scan/vi_gate.py`, split by sha1(video id)); no Video-Index data. The temperature/threshold choice between the two
  `when/count` selection rules was made on the CAL half (count uses `fwer`, `when` uses BH).
- **Gate not met in full** (`video/gate/results/gate_stats.json`, `gate_e2e.json`): `ever` false alarm is within alpha ± 0.02 up to 512
  clips but 0.15 at 900 clips (n=127; the client now caps at 600); the within-video FDR of `when` is 0.30 (0.23 with a
  2-s tolerance) against q = 0.1, because the BH guarantee is relative to clips of other videos, not neighbouring
  clips of the same video. The tool doc therefore claims a guarantee only for `ever` YES, not for `when`. We ran VI
  anyway, as a measured deviation from §9 of `research/one_query_many_images.md`.
- **Mid-run tool change (21:22 UTC).** `vdmscan reps` printed the expected-value count (RepCount-A MAE 0.46); it now
  prints the per-window argmax sum (MAE 0.20, as in the offline eval). One scan-arm item had used `reps` before the
  change (S-EMBER_346). Nothing else in the tool or prompt changed during the run.
- **Scan server outage 21:30–21:58 UTC.** The server process was killed by the VM's host-RAM OOM (banks demoted
  to host RAM plus request bodies reached ~180 GB). Six scan-arm items whose agent saw an infrastructure error (or a
  GPU OOM earlier, before the two-tier cache) were moved to `runs/_compromised/` unscored and are re-run at the end
  of the arm: S-EMBER_2431, 2539, 3231, 346, 544, UrbanVideo-Bench_5348. The server now runs under a restart loop with
  90 GB total / 40 GB GPU bank budgets, and gen-host probes it every 30 s (`runs/_gateway/scan_health.log`).
