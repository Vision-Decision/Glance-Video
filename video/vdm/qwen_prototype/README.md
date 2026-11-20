# vdm prototype (find / count)

Status: prototype, offline test only. Raw **Qwen/Qwen3.5-4B** stands in for our decision model.

## What it does

`vdm` turns a yes/no visual question into a time curve P(t) and reads intervals and counts off it:

1. Decode the video once with ffmpeg at `frames/clip` fps, short side 336 px (multiple of 32).
2. Group the frames into overlapping clips (default 2 s long, 1 s stride, 4 frames each).
3. For each clip, run **one prefill** of Qwen3.5-4B with the k frames as images and the prompt
   *"The images are k frames sampled in order from a 2-second clip of a video. Question: <q> Answer with yes or no."*
   Then P(yes) = sigmoid(logsumexp(logits[yes-variants]) − logsumexp(logits[no-variants])) at the first answer
   position. There is no generation. Clips are batched.
4. `find`: apply a hysteresis threshold to P(t) (enter at `hi`, default 0.5; stay while ≥ 0.6·hi). Each clip stands for
   [centre − stride/2, centre + stride/2], so one low clip separates two events. The output is a list of intervals.
5. `count`: the number of separate intervals.

```bash
# needs torch, transformers>=5, pillow, numpy, matplotlib, ffmpeg on PATH; CUDA, MPS or CPU
python video/vdm/vdm.py find  "Is the person opens the fridge?" --video v.mp4 --png p.png
python video/vdm/vdm.py count "Is a person places a cup on the table?" --video v.mp4 --start 0 --end 300
python video/vdm/run_tests.py --items items.json --videos <dir with videos/> --out video/vdm/results
```

Files: `vdm.py` (CLI, decoding, clip grouping, scan, intervals, plot), `qwen_backend.py` (yes/no scorer; picks
CUDA, then MPS, then CPU), `run_tests.py` (offline test on Video-Index items, kept in the private dataset with its item fixtures; scan questions fixed in the file before
any run), `test_items.json` (the 5 test items), `run_colab.sh` (one command: create `dv-vdm` on G4, run, fetch
`results/`, always stop the VM), `results/` (empty so far).

## Status (2026-11-18): code done, model run blocked

- **Done and checked without the model:** frame decoding and streaming clip grouping (memory stays bounded on a
  2.3 h movie), the batched scorer interface, hysteresis intervals and counts (fake-scorer test on a synthetic
  video), plots, and the option mapping. Also checked with the real Qwen3.5-4B processor: the chat template
  (thinking disabled, so the answer token comes first), yes/no token ids (`Yes` 9175, `yes` 9405, ` Yes` 7179 /
  `No` 2665, `no` 2083, ` No` 2233), and left-padded multi-image batching (448×336 frame = 140 visual tokens,
  4-frame clip ≈ 600 tokens).
- **Not run yet: no P(t) curves, counts or clips/s.**
  1. Colab: `colab-pool new dv-vdm --gpu g4` exits 69 because the primary account is at its 4-session limit
     (dv-27b-2, dv-abl, dv-v1, node-p013, none of which may be touched). The pool rules forbid switching accounts.
  2. Local fallback (M3 Max, MPS): the Qwen3.5-4B weights are not in the local HF cache (only config files).
     Bandwidth was 0.3–0.6 MB/s at the time, so ~9 GB would take about 4 h, and the download competed with the
     sandbox's Whisper download. It was paused. The partial files are in `~/.cache/huggingface/hub` and
     `HF_HUB_DISABLE_XET=1 hf download Qwen/Qwen3.5-4B` resumes them.
- **To finish:** when a slot is free, run `bash video/vdm/run_colab.sh` (~15–30 min of G4 time expected; it stops
  the VM on exit). Or, once the weights are local, run `python video/vdm/run_tests.py --items video/vdm/test_items.json
  --videos data/video_index/v1_1 --out video/vdm/results` (MPS).

## Test plan (fixed before any run)

Items: v1.1 counting questions where Claude Opus 5.5 timed out or at most 6 of 21 agents were right, and where a
visual yes/no scan can in principle answer (not dialogue). Thresholds are reported at 0.3 / 0.5 / 0.7 and are not
tuned on these videos (evaluation-only licence).

| Item | Video | Agents right / Opus 5.5 | Scan question | Options (count) | Key |
|---|---|---|---|---|---|
| S-EMBER_26 | [Video-Index item details redacted] |
| S-EMBER_7718 | [Video-Index item details redacted] |
| S-EMBER_888 | [Video-Index item details redacted] |
| S-EMBER_1448 | [Video-Index item details redacted] |
| ALLVB_9711 | [Video-Index item details redacted] |

Per item the run writes `results/<item>_<scan>.json` (full P(t)), `results/<item>_<scan>.png` and
`results/summary.json` (count at each threshold, the option it implies by nearest count, timing, clips/s).

## Limitations and expected failure modes

- Raw Qwen3.5-4B is not calibrated for yes/no on clips. The thresholds are a priori, and the counts can be off by
  an over- or under-segmented event. This is the main reason for our trained model with calibrated P.
- 4 frames per 2 s clip (2 fps) can miss sub-second actions (a dip of the brush, a spoon lift). Denser clips cost
  linearly more.
- Counting = number of separated intervals. Two events less than one stride apart merge, and one long action with a
  P dip splits. The hysteresis (enter 0.5, stay above 0.3) only partly fixes this.
- Qualifiers in the question ("after the second visit", "in the kitchen scene", "since the door opened") need a
  `find` for the anchor event first, then a `count` restricted with `--start/--end`. In `vdm` the planner does
  this, and the ALLVB test exercises it (scene scan, then count).
- Long movies: a full 2.3 h scan at 2 s stride is ~4,160 clips. Coarse-to-fine (scene scan at 4 s/2 frames, then a
  dense scan only inside the candidate scenes) is the intended use.

## Next steps

1. Run the plan above on G4 and fill in the table (count, implied option, wall time, clips/s).
2. Add `verify` (one clip window, one claim → P) and `ego-motion` (left/right/forward from optical flow plus a yes/no
   check). Make `vdm` a CLI on the sandbox PATH that calls a GPU server, for the +vdm runs in `../design.md`.
3. Swap the backend for our decision model (same `score(clips, question)` interface).
