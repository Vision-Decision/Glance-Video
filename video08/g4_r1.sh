#!/usr/bin/env bash
# R1 cells that need source resolution (not 256x256), run on the G4 (the A100 keeps only the 256 cache).
export GV=/content/gv PYTHONPATH=/content/gv/code HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
cd $GV; step() { echo "$(date +%H:%M:%S) $*" >> logs/progress.log; }
mkdir -p out/r1
[ -f data/idx/pushup.jsonl ] || { python code/video08/data/fetch_all.py $GV/data pushup > logs/fetch_pushup.log 2>&1; python - <<'P'
import glob, json, os, re
rows = []
for p in sorted(glob.glob("/content/gv/data/raw/pushup/*.mp4")):
    m = re.match(r"(.+?)_(\d+)_(.+)_\[(\d+)\]\.mp4$", os.path.basename(p))
    if m: rows.append(dict(path=p, id=f"{m.group(1)}_{m.group(2)}", exercise=m.group(3), count=int(m.group(4))))
open("/content/gv/data/idx/pushup.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows)); print(len(rows))
P
}
Q="python -m video08.evals.m6"; pids=()
for cfg in "2 128" "2 512" "30 128" "30 512" "2 352x192" "2 192x352"; do set -- $cfg
  $Q --bench presence --arm stream --fps $1 --size $2 --n 150 --out out/r1/presence_f$1_$2.jsonl > logs/r1_p_$1_$2.log 2>&1 & pids+=($!); done
for cfg in "2 512" "2 352x192"; do set -- $cfg
  $Q --bench pushup --arm stream --fps $1 --size $2 --n 100 --out out/r1/pushup_f$1_$2.jsonl > logs/r1_u_$1_$2.log 2>&1 & pids+=($!); done
step "r1 launched ${#pids[@]}"
for p in "${pids[@]}"; do wait $p; step "r1 pid $p rc=$?"; done; step R1-DONE
