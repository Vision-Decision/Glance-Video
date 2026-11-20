#!/usr/bin/env bash
# CPU checks for this repo (no GPU, no checkpoint; tiny random Qwen3.5 config).
#  1. glance-engine's scan equivalence test, including the HTTP scan server from video/vdm/gv/ (vdmscan backend).
#  2. Every Python file compiles.
set -euo pipefail
cd "$(dirname "$0")/.."
SCAN_SERVER_DIR="$PWD/video/vdm/gv" python -m scan.test_scan_equivalence
python - <<'PY'
import pathlib, sys
bad = []
for p in pathlib.Path(".").rglob("*.py"):
    if ".venv" in p.parts:
        continue
    try:
        compile(p.read_text(), str(p), "exec")
    except SyntaxError as e:
        bad.append(f"{p}: {e}")
print("\n".join(bad) or "compile check: all files OK")
sys.exit(1 if bad else 0)
PY
