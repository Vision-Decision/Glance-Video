#!/usr/bin/env bash
# Glance-Video records -> mixture -> training run on a Colab VM. Clips at /content/gv/clips, data_root=/content.
# Usage (VM): MODE=pilot|full bash /content/dv/scripts/video/vm_train.sh > /content/gv_train_$MODE.log 2>&1
#   pilot: ~1/4 of the video records + replay, MAX_STEPS steps (default 500)
#   full : all video records + replay, 1 epoch (INIT=/content/init/ckpt.pt must be the v1.1 final if available)
set -uo pipefail
MODE=${MODE:-pilot}; R=/content/gv; D=/content/dv; REC=${REC:-$R/records}
cd $R
if [ ! -f $REC/summary.json ] || [ "${REGEN:-0}" = 1 ]; then
  python $D/scripts/video/gen_records.py --root $R --out $REC ${GEN_ARGS:-} > $R/logs/gen.log 2>&1
fi
tail -60 $REC/summary.json
V="$REC/train_k700.jsonl $REC/train_charades.jsonl $REC/train_countix.jsonl $REC/train_repcount.jsonl"
specs=""
if [ $MODE = pilot ]; then
  for f in $V; do [ -f $f ] && specs="$specs $f:$(( $(wc -l < $f) / 4 ))"; done
  specs="$specs $R/replay/v1_text.jsonl:8000 $R/replay/v1_coco.jsonl:2500"
  EXTRA="max_steps=${MAX_STEPS:-500} out_dir=/content/out/gv0_pilot"
else
  for f in $V; do [ -f $f ] && specs="$specs $f:0"; done
  NV=$(cat $V 2>/dev/null | wc -l)
  specs="$specs $R/replay/v1_text.jsonl:$((NV / 4)) $R/replay/v1_coco.jsonl:$((NV / 12))"
  EXTRA="out_dir=/content/out/gv0"
fi
python $D/scripts/video/make_mix.py $REC/train_mix_$MODE.jsonl $specs
cd $D && python train.py --cfg configs/video_v0.json --set train_file=$REC/train_mix_$MODE.jsonl $EXTRA ${SETS:-}
echo "[train] $(date -u +%T) exit $?"
