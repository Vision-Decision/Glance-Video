#!/usr/bin/env bash
# G4 (RTX PRO 6000) runner v2: setup, then M7 (resumes from any uploaded partial jsonl) + M8 (fixed 4-60 s window) in
# parallel, then the timing runs alone (strict per-frame M8, throughput).
export GV=/content/gv PYTHONPATH=/content/gv/code HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $GV; mkdir -p logs out/m7 out/m8 out/throughput; mkdir -p $HOME/glance-video-exp && ln -sfn $GV/data $HOME/glance-video-exp/data
step() { echo "$(date +%H:%M:%S) $*" >> logs/progress.log; }
step setup
pip -q install transformers==5.23.0 av flash-linear-attention > logs/pip.log 2>&1; step "pip rc=$?"
timeout 1200 pip -q install causal-conv1d --no-build-isolation > logs/pip_cc.log 2>&1 &
CC=$!
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-0.8B')" > logs/model.log 2>&1; step "model rc=$?"
if [ ! -f data/raw/charades/.done ]; then
  mkdir -p data/raw/charades
  python code/video08/data/pdl.py https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip data/raw/charades.zip 24 > logs/pdl.log 2>&1
  (cd data/raw/charades && xargs -a $GV/data/eval_ids.txt unzip -q ../charades.zip) && touch data/raw/charades/.done && rm -f data/raw/charades.zip
fi
step "charades test videos: $(ls data/raw/charades/Charades_v1_480 | wc -l)"
wait $CC; step "causal_conv1d rc=$?"
Q="python -m video08.evals"; pids=()
for kv in layered short; do for f in 2 30; do
  $Q.m7 --fps $f --kv $kv --trials 60 --out out/m7/${kv}_fps$f.jsonl > logs/m7_${kv}_$f.log 2>&1 & pids+=($!); done; done
$Q.m7 --fps 2 --kv full --trials 60 --out out/m7/full_fps2.jsonl > logs/m7_full_2.log 2>&1 & pids+=($!)
for f in 2 30; do for arm in long fresh; do
  $Q.m8 --fps $f --minutes 60 --arm $arm --out out/m8/${arm}_fps$f.json > logs/m8_${arm}_$f.log 2>&1 & pids+=($!); done; done
step "launched ${#pids[@]}"
for p in "${pids[@]}"; do wait $p; step "pid $p rc=$?"; done
$Q.m8 --fps 30 --minutes 10 --chunk 1 --arm long --probe_every 60 --out out/m8/strict_fps30_10min.json > logs/m8_strict.log 2>&1; step "m8 strict rc=$?"
$Q.throughput --out out/throughput/gen_fps30_g4_v2.json > logs/throughput.log 2>&1; step "throughput rc=$?"
step ALL-DONE
