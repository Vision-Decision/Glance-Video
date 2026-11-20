#!/bin/bash
# keep the scan server up (restart on crash; banks are rebuilt on demand by the client)
cd /content/dv
while true; do
  echo "$(date -u) start" >> /content/serve_loop.log
  PYTHONPATH=/content/dv SCAN_MAX_BANK_GB=150 SCAN_GPU_BANK_GB=70 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python video/vdm/gv/scan_server.py --ckpt /content/ckpt/scan/short_gv0/ckpt_final.pt --calib /content/CALIB.json \
    --readout short --port 8000 >> /content/scan_server.log 2>&1
  echo "$(date -u) exited $?" >> /content/serve_loop.log; sleep 3
done
