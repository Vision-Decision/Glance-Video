"""Write the reproducibility files that upload_loop.sh pushes with out/ to HF Vision-Decision/glance-train-ckpts:video08/:
docs (README, PREREG, P4_DESIGN), one README.md per experiment directory, and a data manifest (video ids used for
training / evaluation, dedup + exclusion rules in words, decode-cache parameters).
Nothing Video-Index related is written: no VI videos, questions, per-item results or id lists. (The Charades ids that
Video-Index sources use are excluded from training; that list itself is NOT uploaded, only the rule.)
python -m video08.hf_manifest OUT DATA"""
import json, os, shutil, subprocess, sys

OUT, DATA = sys.argv[1], sys.argv[2]
CODE = os.path.dirname(os.path.abspath(__file__))
try:
    commit = open(f"{CODE}/.commit").read().strip()
except OSError:
    commit = "unknown"
HW = "a100 host: 4x A100 80G shared through the training agent's queue (~/glance-queue); Colab G4 (RTX PRO 6000 96G) for M7/M8 and daytime S1"
COMMON = f"code: Vision-Decision/glance-video branch `video-08b`, commit `{commit}`, directory `video08/`\nhardware: {HW}\nmodel: Qwen/Qwen3.5-0.8B frozen (bf16), stream = 64 tokens/group (256x256), KV sink + 4 s full + 4-60 s @2 fps, GDN state carried\n"
EXP = {
    "m6": ("M6: zero-shot fps=2 vs fps=30 (and native / R1 grid arms)", "seed 0 (video shuffle), batched lockstep streams (GV_BATCH=8)",
           "video08/q.sh evals.m6 --bench {presence,sta,pushup} --arm {stream,native} --fps F --n N --out m6/<bench>_<arm>_fps<F>.jsonl"),
    "r1": ("R1: zero-shot robustness grid (fps x tokens per group x aspect)", "seed 0, same video subset as M6 (first 150 / 100)",
           "video08/q.sh evals.m6 --bench B --arm stream --fps F --size S|WxH --n N --out r1/..."),
    "deploy": ("Deployment protocol streams (Charades test, 10-min streams, 16 rules, readout every 0.5 s) + question-free features", "stream seed s (0..11)",
               "video08/q.sh evals.deploy --split eval --fps F --arm {stream,recent4} --streams 12 --minutes 10 --rules 16 --out deploy"),
    "deploy_train": ("Probe-training feature streams (Charades train split)", "stream seed s", "video08/q.sh evals.deploy --split train --arm none ..."),
    "p3": ("P3 / S1: frozen backbone, readout heads (S1a) and + question-block LoRA r=16 (S1b); evals head vs generative", "torch seed 0; data order from worker rng",
           "video08/q.sh train_s1 --out p3/<run> --steps 1200 --batch 4 --qbatch 8 --times 2 --write_s 2 [--lora 16] --workers 6; then evals.p3_eval --ckpt ... --fps {2,30}"),
    "m7": ("M7: GDN memory span with inserted 1 s word cards (Charades test background)", "trial t uses rng seed t (same videos/words for every arm)",
           "python -m video08.evals.m7 --fps {2,30} --kv {layered,short,full} --trials 60"),
    "m8": ("M8: 60-min stream stability (continuous vs fresh per probe); strict per-frame timing", "seed 0", "python -m video08.evals.m8 --fps F --minutes 60 --arm {long,fresh}"),
}
os.makedirs(f"{OUT}/docs", exist_ok=True)
for f in ("README.md", "PREREG.md", "P4_DESIGN.md"):
    if os.path.exists(f"{CODE}/{f}"):
        shutil.copy(f"{CODE}/{f}", f"{OUT}/docs/{f}")
for d, (what, seed, cmd) in EXP.items():
    if os.path.isdir(f"{OUT}/{d}"):
        open(f"{OUT}/{d}/README.md", "w").write(f"# {d}\n\n{what}\n\n{COMMON}seed: {seed}\nlaunch: `{cmd}`\n\n"
                                                 "Files: per-item jsonl / npz results; for training runs log.jsonl, config.json, ckpt_step*.pt "
                                                 "(readout + LoRA weights), ckpt_last.pt (with optimizer), ckpt_final.pt.\n")
# data manifest (ids only for the datasets we use; no Video-Index material)
M = f"{OUT}/data_manifest"; os.makedirs(M, exist_ok=True)
sp = json.load(open(f"{DATA}/idx/splits.json"))
for k in ("train", "probe_val", "eval"):
    open(f"{M}/charades_{k}_ids.txt", "w").write("\n".join(sp[k]) + "\n")
for s in ("train", "validation", "test"):
    f = f"{DATA}/idx/repcount_{s}.jsonl"
    if os.path.exists(f):
        rows = [json.loads(l) for l in open(f)]
        open(f"{M}/repcount_{s}_ids_used.txt", "w").write("\n".join(r["id"] for r in rows if not (s == "train" and r["vi_contaminated"])) + "\n")
if os.path.exists(f"{DATA}/idx/pushup.jsonl"):
    open(f"{M}/pushupbench_ids.txt", "w").write("\n".join(json.loads(l)["id"] for l in open(f"{DATA}/idx/pushup.jsonl")) + "\n")
open(f"{M}/README.md", "w").write(f"""# Data manifest (video08)

{COMMON}
| Set | Source | Use | Ids |
|---|---|---|---|
| Charades v1 480p + Charades-STA | AI2 S3 `Charades_v1_480.zip`; TALL STA annotations | train split minus excluded ids -> S1 training, probe training; 300 held-out train ids -> probe_val; test split -> evaluation only | `charades_*_ids.txt` |
| RepCount-A (LLSP) | hf-mirror `lmms-lab-eval/repcounta-lance` | train (minus pHash-flagged clips) -> count head; test -> evaluation only | `repcount_*_ids_used.txt` |
| PushupBench | hf-mirror `anonymousatom/pushupbench` | evaluation only | `pushupbench_ids.txt` |
| HC-STVG v2, MeViS, LaSOT (6 classes, 10 fps), RefCOCO | HF copies (B group) | B group (boxes); VidSTG evaluation only | added when used |

Dedup / exclusion rules
- Train and evaluation sets are split by video id; no id is in both (asserted by `video08/data/build.py`).
- Video-Index: no Video-Index video or question is used for training, tuning, or uploaded. Charades ids used by any
  Video-Index source benchmark (FAVOR-Bench, TOC-Bench, MVBench, TVBench, TemporalBench, E.T. Bench) and RepCount /
  Countix clips whose 1-fps pHash matches a Video-Index frame are removed from training. The excluded-id lists
  themselves are Video-Index material and are not part of this upload.
- VidSTG (VidOR videos, a Video-Index source via NExT-QA) is never used for training or tuning; evaluation only.
- Ref-YouTube-VOS not used.

Decode cache (`video08/data/cache.py`)
- `cache256/`: every source video once as 256x256 H.264 (libx264 CRF 17, preset veryfast, tune fastdecode, GOP 30,
  original timestamps). `vio.decode` reads it for 256x256 requests.
- `cache_fps2/`: uint8 frames (T,3,256,256) on the exact k/2 s grid + timestamps (.t.npy) for the fps-2 evaluation arms.
- Frames for a target fps = latest source frame at each k/fps (repeats when the source is slower; Charades is 24/25 fps,
  so a 30 fps grid repeats ~20% of frames).
""")
print("manifest written")
