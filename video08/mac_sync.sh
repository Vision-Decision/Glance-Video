#!/bin/zsh
# Mac relay (the a100 host has no outbound access): every 15 min pull ~/glance-video-exp/out over the LAN, refresh the
# per-experiment READMEs / data manifest on the host (offline), upload to HF Vision-Decision/glance-train-ckpts:video08/,
# then delete the local copy. Nothing Video-Index related exists in out/ (see video08/hf_manifest.py).
L=${TMPDIR:-/tmp}/gv_relay
while true; do
  ssh a100 '. ~/glance-video-exp/env.sh; cd $GV && $PY -m video08.hf_manifest $GV/out $GV/data' >/dev/null 2>&1
  rm -rf $L; mkdir -p $L
  if rsync -a --exclude '*.tmp' --exclude 'smoke/' --exclude '_hf/' a100:glance-video-exp/out/ $L/out/; then
    (cd /tmp && uvx --from huggingface_hub python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path='$L/out', path_in_repo='video08', repo_id='Vision-Decision/glance-train-ckpts', commit_message='video08: periodic sync (Mac relay)')
print('ok')" 2>&1 | tail -1) && echo "$(date +%H:%M) synced $(du -sh $L/out | cut -f1)"
  else echo "$(date +%H:%M) rsync failed"; fi
  rm -rf $L
  sleep 900
done
