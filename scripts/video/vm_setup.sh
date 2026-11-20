#!/usr/bin/env bash
# Glance-Video Colab VM setup (idempotent). Code must already be at /content/dv (scripts/sync.sh SESSION --code-only).
# Usage (VM): bash /content/dv/scripts/video/vm_setup.sh [clips|noclips] > /content/gv_setup.log 2>&1
# - deps (same pins as scripts/colab_setup.sh) + PyAV
# - Qwen3.5-4B weights
# - Glance-Video shards from the PRIVATE HF dataset Vision-Decision/glance-video-v0 -> /content/gv (clips + records + replay)
# - COCO train2014 (for the v1.1 replay box records) -> /content/coco/train2014
# data_root for training / eval = /content
# HF auth: the VM's HF token (Colab secret / env), never written to files here.
set -uo pipefail
export PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_DISABLE_PROGRESS_BARS=1
MODE=${1:-clips}
echo "[gv-setup] $(date -u) start"
nvidia-smi --query-gpu=name,memory.total --format=csv; nproc; df -h /content | tail -1
pip uninstall -y -q torchao || true
which aria2c >/dev/null || apt-get install -y -qq aria2 >/dev/null 2>&1
pip install -q "transformers==5.23.0" "flash-linear-attention==0.6.0" "peft>=0.17" accelerate pyarrow pandas av opencv-python-headless
python - <<'EOF'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3.5-4B", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.py"])
print("weights", p)
EOF
mkdir -p /content/gv
if [ "$MODE" = clips ]; then
python - <<'EOF'
import os, tarfile, glob
from huggingface_hub import snapshot_download
d = snapshot_download("Vision-Decision/glance-video-v0", repo_type="dataset", local_dir="/content/hf_gv", max_workers=8)
for t in sorted(glob.glob(f"{d}/shards/*.tar")):
    mark = t + ".done"
    if not os.path.exists(mark):
        with tarfile.open(t) as tf:
            tf.extractall("/content", filter="data")
        open(mark, "w").close()
os.system("cp -r /content/hf_gv/records /content/hf_gv/replay /content/gv/ 2>/dev/null; ls /content/gv; du -sh /content/gv")
EOF
fi
if [ ! -f /content/coco/train2014/.done ]; then
  mkdir -p /content/coco && cd /content/coco && (aria2c -q -x 16 -s 16 http://images.cocodataset.org/zips/train2014.zip || wget -q http://images.cocodataset.org/zips/train2014.zip) \
    && unzip -q -o train2014.zip && rm -f train2014.zip && touch train2014/.done
fi
echo "[gv-setup] coco: $(ls /content/coco/train2014 | wc -l)"
echo "[gv-setup] $(date -u) DONE"
