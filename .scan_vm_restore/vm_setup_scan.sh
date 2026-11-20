#!/usr/bin/env bash
# scan server VM setup (idempotent): deps, Qwen3.5-4B, short-readout Glance-Video ckpt, Charades pilot scores
set -uo pipefail
export PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p /root/.cache/huggingface && [ -f /content/hftok ] && install -m 600 /content/hftok /root/.cache/huggingface/token && rm -f /content/hftok
mkdir -p /content/dv && tar xzf /content/dv.tgz -C /content/dv
echo "[setup] $(date -u) start"; nvidia-smi --query-gpu=name,memory.total --format=csv
pip uninstall -y -q torchao || true
pip install -q "transformers==5.23.0" "flash-linear-attention==0.6.0" "peft>=0.17" accelerate av
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
print(snapshot_download("Qwen/Qwen3.5-4B", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.py"]))
for f in ("scan/short_gv0/ckpt_final.pt", "results/gv0/scan/pilot_clips_shortgv.json.gz", "results/gv0/scan/pilot_clips_gvfinal.json.gz"):
    print(hf_hub_download("Vision-Decision/glance-train-ckpts", f, local_dir="/content/ckpt"))
PY
echo "[setup] $(date -u) DONE"
