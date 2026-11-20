#!/usr/bin/env bash
# One-shot: run the vdm offline test on ONE Colab G4 VM named dv-vdm, fetch results, always stop the VM.
#   bash video/vdm/run_colab.sh            # all test items
#   ONLY=S-EMBER_888,S-EMBER_1448 bash video/vdm/run_colab.sh
# Needs a free slot on the pool's primary account (exit 69 = full: do not switch accounts, retry later).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NAME=dv-vdm
OUT="$HERE/results"
mkdir -p "$OUT"
trap 'timeout 300 colab-pool stop "$NAME" || true' EXIT

timeout 900 colab-pool new --if-missing "$NAME" --gpu g4
for f in vdm.py qwen_backend.py run_tests.py test_items.json; do
  timeout 300 colab-pool upload "$NAME" "$HERE/$f" "/content/vdm/$f"
done
cat > /tmp/vdm_setup_$$.py <<'EOF'
import subprocess
def run(c, to):
    r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=to)
    print(c[:70], "rc", r.returncode, r.stdout[-300:], r.stderr[-300:], flush=True)
run("nvidia-smi --query-gpu=name,memory.total --format=csv", 60)
run("pip install -q -U 'transformers>=5' accelerate huggingface_hub matplotlib", 900)
run("hf download Qwen/Qwen3.5-4B > /dev/null 2>&1; echo ok", 1800)
EOF
timeout 3600 colab-pool exec "$NAME" -f /tmp/vdm_setup_$$.py
cat > /tmp/vdm_run_$$.py <<EOF
import subprocess, sys
r = subprocess.run([sys.executable, "/content/vdm/run_tests.py", "--items", "/content/vdm/test_items.json",
                    "--videos", "/content/vi", "--out", "/content/vdm/results", "--only", "${ONLY:-}"],
                   capture_output=True, text=True, timeout=7000)
print(r.stdout[-4000:]); print(r.stderr[-2000:])
subprocess.run("cd /content/vdm && tar czf results.tgz results", shell=True, timeout=300)
EOF
timeout 7500 colab-pool exec "$NAME" -f /tmp/vdm_run_$$.py | tee "$OUT/colab_log.txt"
timeout 600 colab-pool download "$NAME" /content/vdm/results.tgz "$OUT/results.tgz"
tar xzf "$OUT/results.tgz" -C "$HERE" && rm "$OUT/results.tgz"
rm -f /tmp/vdm_setup_$$.py /tmp/vdm_run_$$.py
