#!/usr/bin/env bash
# CPU job: push ~/glance-video-exp/out to HF Vision-Decision/glance-train-ckpts:video08/ every 15 min (token only from ~/.hf.env)
. $HOME/glance-video-exp/env.sh
set -a; . $HOME/.hf.env; set +a
while true; do
  $PY -m video08.hf_manifest $GV/out $GV/data > /dev/null 2>&1
  $PY - <<'PYEOF'
import os
from huggingface_hub import HfApi
api = HfApi(); out = os.path.expanduser("~/glance-video-exp/out")
try:
    api.upload_folder(folder_path=out, path_in_repo="video08", repo_id="Vision-Decision/glance-train-ckpts",
                      commit_message="video08: periodic sync of results / checkpoints", ignore_patterns=["*.tmp", "_hf/*", "**/smoke/*"])
    print("synced", flush=True)
except Exception as e:
    print("sync failed", repr(e)[:300], flush=True)
PYEOF
  sleep 900
done
