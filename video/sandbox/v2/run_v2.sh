#!/bin/bash
# paired VI runs v2: DeepSeek V4.1 Flash (official API via vi-gateway-v2, 40-step budget), both arms at once, same order.
# usage: run_v2.sh RUN [run.py args...]   e.g. run_v2.sh v2_pilot --ids a,b,c   |   run_v2.sh v2_hard --set hard60
cd ~/videoindex/v2/sandbox
export VI_DATA=~/videoindex/data VI_RESEARCH=~/videoindex/research VI_VDMSCAN=~/videoindex/v2/vdmscan \
       VI_NET=vi-internal-v2 VI_GW=vi-gateway-v2 VI_SCAN_URL=http://127.0.0.1:18084
RUN=$1; shift
exec python3 run.py --run $RUN --arms base,scan --warm --transcripts $VI_DATA/transcripts --max-steps 40 \
     --concurrency ${CONC:-4} --cpus 2 --memory 24g --timeout 1800 "$@"
