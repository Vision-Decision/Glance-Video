---
license: other
tags: [video, research-only, private]
---
# glance-video-v0 (PRIVATE, research use only)

Training clips and typed-question records for Glance-Video (clip-level decision model behind `vdm` v1).
Code: `scripts/video/` in the decision-vison repo. Not for redistribution: several sources forbid it.

## Contents
- `shards/<set>-NNNN.tar`: normalised clips (8 fps, short side 256, H.264 CRF 26, no audio) under `gv/clips/<set>/`,
  each shard with the set's `index.jsonl`.
- `records/`: `train_*`, `calib_*` (temperature fitting), `eval_*` (held out) typed-question records
  (format: `dv/data/video.py`); `summary.json` = counts per task.
- `replay/`: a sample of the Glance v1.1 training records (text + COCO box) mixed in against forgetting.
- `meta/`: annotations used by the generators, and the decontamination files (`decon/exclusions.json`, VI frame
  hashes, VATEX id lists, raw-file SHA-256).

## Sources and licences
| Set | Source | Licence / terms | Use here |
|---|---|---|---|
| k700val | Kinetics-700-2020 val (CVDF S3) | annotations CC BY 4.0; videos are YouTube content | self-supervised (arrow of time, camera motion, speed, freeze, order, synthetic repetition), action yes/no; Countix-val counts |
| countix_train | Kinetics-700-2020 train clips listed in Countix train | Kinetics as above; Countix (DeepMind) ships without a licence file: research only | repetition counts |
| charades | Charades v1 480p (AI2) + Charades-STA (TALL) | AI2 "License for Non-Commercial Use": academic/non-profit research, no redistribution | events with temporal hard negatives, reversal antonyms, order; STA sentences; test split = eval only |
| repcount_* | RepCount-A (LLSP/TransRAC) via HF mirror lmms-lab-eval/repcounta-lance | no dataset licence stated; academic use | train/val: windowed counts; test: eval only |
| replay | Glance v1.1 mixture (see the Glance model card) | per source (incl. non-commercial ones) | forgetting control |

Not used (manual steps for a human, never accepted on the user's behalf): Something-Something v2 (Qualcomm login +
licence), official ActivityNet videos (Google form), official RepCount-A OneDrive (password). Excluded on purpose:
NExT-QA, Perception Test, STAR/AGQA, Ego4D/EgoSchema, MSR-VTT, VATEX, YouCook2 (Video-Index sources or reused by them).

## Decontamination against Video-Index v1.0 + v1.1
See `meta/decon/exclusions.json`: dataset-level exclusions, exact ids (v1.1 `source_video`, all Charades ids used by
FAVOR-Bench / TOC-Bench / MVBench / TVBench / E.T. Bench, VATEX val+test YouTube ids), raw-file SHA-256 vs VI video
ids, and 1-fps pHash matches of every clip against all VI videos. Video-Index videos themselves are evaluation-only and
are not in this dataset (only their frame hashes).
