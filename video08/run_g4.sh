#!/usr/bin/env bash
# G4 (RTX PRO 6000) runner for the long-stream P2 inference: setup, split-attention check, timing runs alone, then M7/M8.
export GV=/content/gv PYTHONPATH=/content/gv/code HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $GV; mkdir -p logs out/m7 out/m8 out/throughput
step() { echo "$(date +%H:%M:%S) $*" >> logs/progress.log; }
step setup
mkdir -p $HOME/glance-video-exp && ln -sfn $GV/data $HOME/glance-video-exp/data   # scripts default to ~/glance-video-exp/data
pip -q install transformers==5.23.0 av flash-linear-attention > logs/pip.log 2>&1; step "pip rc=$?"
timeout 1200 pip -q install causal-conv1d --no-build-isolation > logs/pip_cc.log 2>&1; step "causal_conv1d rc=$? $(python -c 'import causal_conv1d;print(causal_conv1d.__version__)' 2>&1 | tail -1)"
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-0.8B')" > logs/model.log 2>&1; step "model rc=$?"
if [ ! -f data/raw/charades/.done ]; then
  mkdir -p data/raw/charades
  python code/video08/data/pdl.py https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip data/raw/charades.zip 24 > logs/pdl.log 2>&1
  (cd data/raw/charades && xargs -a $GV/data/eval_ids.txt unzip -q ../charades.zip) && touch data/raw/charades/.done && rm -f data/raw/charades.zip
fi
step "charades test videos: $(ls data/raw/charades/Charades_v1_480 | wc -l)"
V=$(ls data/raw/charades/Charades_v1_480/*.mp4 | head -1)
python -m video08.tests.test_split $V > logs/test_split.log 2>&1; step "test_split rc=$? $(tail -1 logs/test_split.log)"
A=$(python - <<'P'
import re
s=open("/content/gv/logs/test_split.log").read().strip().splitlines()[-1]
m=re.findall(r"[-0-9.e]+", s); print("split" if len(m)>=2 and float(m[0])<1.0 and float(m[1])<0.03 else "sdpa")
P
)
export GV_ATTN=$A; step "attention impl: $A"
Q="python -m video08.evals"
# timing runs alone on the GPU
$Q.throughput --out out/throughput/gen_fps30_g4.json > logs/throughput.log 2>&1; step "throughput rc=$?"
$Q.m8 --fps 30 --minutes 10 --chunk 1 --arm long --probe_every 60 --out out/m8/strict_fps30_10min.json > logs/m8_strict.log 2>&1; step "m8 strict rc=$?"
# everything else in parallel
pids=()
for kv in layered short; do for f in 2 30; do
  $Q.m7 --fps $f --kv $kv --trials 60 --out out/m7/${kv}_fps$f.jsonl > logs/m7_${kv}_$f.log 2>&1 & pids+=($!); done; done
$Q.m7 --fps 2 --kv full --trials 60 --out out/m7/full_fps2.jsonl > logs/m7_full_2.log 2>&1 & pids+=($!)
$Q.m8 --fps 30 --minutes 60 --arm long --out out/m8/long_fps30.json > logs/m8_long30.log 2>&1 & pids+=($!)
$Q.m8 --fps 30 --minutes 60 --arm fresh --out out/m8/fresh_fps30.json > logs/m8_fresh30.log 2>&1 & pids+=($!)
$Q.m8 --fps 2 --minutes 60 --arm long --out out/m8/long_fps2.json > logs/m8_long2.log 2>&1 & pids+=($!)
$Q.m8 --fps 2 --minutes 60 --arm fresh --out out/m8/fresh_fps2.json > logs/m8_fresh2.log 2>&1 & pids+=($!)
step "launched ${#pids[@]}"
for p in "${pids[@]}"; do wait $p; step "pid $p rc=$?"; done
step ALL-DONE
