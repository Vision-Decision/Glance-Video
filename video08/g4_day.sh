#!/usr/bin/env bash
# Daytime continuation on the G4: resume S1a / S1b from the A100's ckpt_last (same args), then P3 eval of both arms.
# Head readout and the generative baseline run in the same p3_eval process -> same hardware for the paired comparison.
export GV=/content/gv PYTHONPATH=/content/gv/code HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $GV; step() { echo "$(date +%H:%M:%S) $*" >> logs/progress.log; }
while [ ! -f logs/progress.log ] || ! grep -q G4-PREP-DONE logs/progress.log; do sleep 30; done
O=$GV/out/p3; pids=()
python -m video08.train_s1 --out $O/s1a --steps 1200 --batch 4 --qbatch 8 --times 2 --write_s 2 --workers 8 > logs/s1a.log 2>&1 & pids+=($!)
python -m video08.train_s1 --out $O/s1b_lora16 --steps 1200 --batch 4 --qbatch 8 --times 2 --write_s 2 --lora 16 --workers 8 > logs/s1b.log 2>&1 & pids+=($!)
step "s1 resume launched"
for p in "${pids[@]}"; do wait $p; step "s1 pid $p rc=$?"; done
for r in s1a s1b_lora16; do for f in 2 30; do
  [ -f $O/$r/ckpt_final.pt ] && python -m video08.evals.p3_eval --ckpt $O/$r/ckpt_final.pt --fps $f --n 300 --out $O/eval_${r}_fps$f.jsonl > logs/p3_${r}_$f.log 2>&1 &
done; done; wait; step P3-EVAL-DONE
