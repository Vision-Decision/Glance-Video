#!/bin/bash
# paired VI runs v2 with Claude Opus 5.5 (Claude subscription OAuth via --env-file; HTTPS through the gateway's CONNECT
# allowlist api.anthropic.com:443), both arms at once, 1 agent per arm (subscription rate limits). The gateway cannot
# see TLS traffic, so no step/time budget injection: the 30-min hard timeout is the only limit (same for both arms).
cd ~/videoindex/v2/sandbox
unset ANTHROPIC_API_KEY
export VI_DATA=~/videoindex/data VI_RESEARCH=~/videoindex/research VI_VDMSCAN=~/videoindex/v2/vdmscan \
       VI_NET=vi-internal-v2o VI_GW=vi-gateway-v2o VI_SCAN_URL=http://127.0.0.1:18084
RUN=$1; shift
exec python3 run.py --run $RUN --backend opus --model claude-opus-5-5 --env-file ~/.claude-bench/bench.env --arms base,scan --warm --time-note \
     --transcripts $VI_DATA/transcripts --concurrency ${CONC:-1} --cpus 2 --memory 24g --timeout 1800 "$@"
