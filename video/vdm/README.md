# vdm v0 — Glance-4B as a per-frame visual detector for video agents

`vdm` gives a video agent one extra CLI. It samples frames with ffmpeg, scores them with **Glance-4B-preview**
(our single-prefill decision model: box presence gate, yes/no head, choice head) on a GPU server, and turns the
per-frame scores into time intervals, counts, verifications, boxes and option scores. Camera motion comes from
OpenCV optical flow with no model.

## Pieces

| File | Where it runs | What it does |
|---|---|---|
| `vdm` | inside the agent sandbox (stdlib + numpy + OpenCV + Pillow, Python 3.11) | CLI: `find`, `count`, `verify`, `locate`, `motion`, `choose`; extracts frames (short side 448, JPEG), caches them under `scratch/vdm_cache/`, POSTs to `$VDM_URL/score` in chunks of 48 |
| `server.py` | Colab G4 (RTX PRO 6000) | Glance loaded once; `GET /health`, `POST /score`; batched multi-image version of `Glance.ask` (all questions about one frame from one prefill), LRU cache keyed by (image sha1, question, visual-token budget); `--selftest` checks the batched path against `Glance.ask` |
| `rtunnel_setup.py`, `relay.py` | Colab VM + gen-host | **current**: Colab :8000 → direct `ssh -R` to gen-host 127.0.0.1:18080 with a key restricted to that one forward → relay on docker0 **172.17.0.1:18080** |
| `tunnel.sh`, `tunnel_loop.sh` | laptop | fallback only: Colab → laptop (`colab-pool ssh --proxy-mode`) → gen-host. It broke after ~1 h and its reconnects held the pool lock exclusively for minutes, blocking other agents' pool commands — do not use while others share the pool |
| `TOOL.md` | — | the 15-line description appended to the agent's system prompt |
| `choose_all.py` (offline tests and yes/no ablation: private dataset, `glance-video/video/vdm/`) | gen-host | choose-only lower bound |
| `results/` | — | outputs of the above |
| `qwen_prototype/` | — | the earlier clip-level Qwen3.5-4B prototype (never run; kept for reference) |

## Deploy

```bash
colab-pool new dv-vdm --gpu g4 --account acct-7 --allow-other-account   # user approved a second account
# on the VM: pip install "glance @ git+https://github.com/Vision-Decision/Glance" flash-linear-attention
#            pip uninstall -y torchao   (Colab's torchao 0.10 breaks peft; uninstall with `python -m pip` too)
#            python server.py --port 8000 --batch 32 --selftest
colab-pool exec dv-vdm -f video/vdm/rtunnel_setup.py   # prints the tunnel key; authorize it once on gen-host (see file)
ssh gen-host 'cd ~/videoindex/vdm && nohup python3 relay.py 172.17.0.1 18080 127.0.0.1 18080 > relay.log 2>&1 &'
# containers on gen-host's default bridge: VDM_URL=http://172.17.0.1:18080 (the sandbox gateway proxies /vdm/*)
# when done: stop dv-vdm and delete the `vdm-colab-dv-vdm` line from gen-host ~/.ssh/authorized_keys
```

Gotcha met twice: `pkill -f <pattern>` inside an `ssh host "..."` or `subprocess(shell=True)` command kills the
shell that carries the pattern; kill in a separate call or use the `[x]yz` bracket trick.

## Speed (G4, bf16, `fla` kernels, 640 visual tokens per frame)

- Server: 64 frames in 3.55 s (**~18 frames/s**); without `fla` ~10–15 frames/s. GPU-bound at 99%.
- End to end inside the agent image (2 CPUs, frame extraction + transfer through two ssh hops): 2–3× the server
  time. 10-minute egocentric video at 2 fps (1,134–1,200 frames): 160–225 s wall. 2.3-hour film at the 1,500-frame
  cap: 645 s wall (seeking 1,500 timestamps on 2 CPUs dominates).

## Offline tests (pre-registered queries, mode `both` = mean of P(present) and P(yes))

| Item | Key | vdm result | Verdict |
|---|---|---|---|
| S-EMBER_26 | [Video-Index item details redacted] |
| S-EMBER_888 | [Video-Index item details redacted] |
| S-EMBER_1448 | [Video-Index item details redacted] |
| S-EMBER_7718 | [Video-Index item details redacted] |
| UrbanVideo-Bench_53 | [Video-Index item details redacted] |
| ALLVB_9711 | [Video-Index item details redacted] |

**Post-hoc ablation** (chosen after seeing the table above, so not a clean test): the yes/no head only, with the
action phrased as a question ("Is a person places a cup on the table right now?"), `--mode yesno`:
[Video-Index item details redacted] (S-EMBER_26, S-EMBER_888, S-EMBER_1448)
[Video-Index item details redacted] (S-EMBER_7718)

**Choose-only lower bound** (`vdm choose`, 32 uniform frames, each frame answers the MCQ alone, mean of the option
probabilities; no LLM, no audio): **3/144 = 2.1%**. Chance is 35.7 items (24.8%). The mean probability on the key is
0.093, against 0.248 at chance. By group: perception 1/33, temporal 1/35, spatial 0/39, reasoning 1/37. Mean 14 s per
item. A full end-to-end check on a synthetic video (red circle, answer C) returns C at 0.999, and the server picks
the correct option at positions A, C, E and B, so this is not a label-mapping bug. It is a selection effect: v1.1
kept only items that frontier agents, who look at frames, got wrong, so single-frame evidence points to a distractor.
The key is also often the last option (41%), which Glance rarely picks (13/144), so predictions often land on the
letter just before the key. Per-item rows are in the private dataset (`glance-video/video/vdm/results/choose_only.jsonl`).

## What this means

- Glance's presence gate is an **object** detector on single frames. "a person places a cup on the table" is judged
  present whenever hand and box are visible, so `count` of actions overcounts badly. Counting actions needs either a
  clip-level (multi-frame) yes/no model — the plan's D5 short-clip scan — or a state-change formulation (count
  transitions of a visible state).
- `find` is useful as a fast "where is X visible" index (landmarks, objects, people) and for `locate`; agents still
  have to look at the frames it points to.
- Video frames are off the calibration distribution: the probabilities rank well but are not calibrated here.

## Protocol deviation (for maintainers)

The +vdm runs add one network endpoint to the sandbox: the gateway forwards `/vdm/*` to the Colab GPU through the
tunnel. The agent still has no internet. The GPU time is inside the agent's 3600 s wall clock. Videos and frames
are only scored, never stored server-side beyond an in-memory cache, and never used for training or calibration.
