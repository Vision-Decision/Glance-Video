#!/usr/bin/env bash
# M8 again with the fixed 4-60 s window (stream grids that start off the 0.5 s lattice kept only 4 s before)
export GV=/content/gv PYTHONPATH=/content/gv/code HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
cd $GV; mkdir -p out/m8_v1_buggywindow; mv out/m8/*.json out/m8_v1_buggywindow/ 2>/dev/null
step() { echo "$(date +%H:%M:%S) $*" >> logs/progress.log; }
Q="python -m video08.evals"; pids=()
$Q.m8 --fps 30 --minutes 60 --arm long --out out/m8/long_fps30.json > logs/m8b_long30.log 2>&1 & pids+=($!)
$Q.m8 --fps 30 --minutes 60 --arm fresh --out out/m8/fresh_fps30.json > logs/m8b_fresh30.log 2>&1 & pids+=($!)
$Q.m8 --fps 2 --minutes 60 --arm long --out out/m8/long_fps2.json > logs/m8b_long2.log 2>&1 & pids+=($!)
$Q.m8 --fps 2 --minutes 60 --arm fresh --out out/m8/fresh_fps2.json > logs/m8b_fresh2.log 2>&1 & pids+=($!)
step "m8 rerun launched"
for p in "${pids[@]}"; do wait $p; step "m8b pid $p rc=$?"; done
while pgrep -f "video08.evals.m7" > /dev/null; do sleep 30; done   # strict timing needs the GPU alone
$Q.m8 --fps 30 --minutes 10 --chunk 1 --arm long --probe_every 60 --out out/m8/strict_fps30_10min.json > logs/m8b_strict.log 2>&1; step "m8b strict rc=$?"
$Q.throughput --out out/throughput/gen_fps30_g4_v2.json > logs/throughput2.log 2>&1; step "throughput2 rc=$?"
step M8-RERUN-DONE
