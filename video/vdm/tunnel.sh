#!/usr/bin/env bash
# vdm tunnel: Colab VM (server :8000) -> laptop :LP -> gen-host 127.0.0.1:RP -> relay on docker0 172.17.0.1:RP
#   bash video/vdm/tunnel.sh [SESSION] [LP] [RP]    defaults: dv-vdm 18000 18080
#   bash video/vdm/tunnel.sh stop
# Needs the laptop awake and online (it carries both ssh legs). Containers on the default docker bridge of gen-host
# reach the server at http://172.17.0.1:RP .
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ "${1:-}" = stop ]; then pkill -f tunnel_loop.sh; pkill -f "localhost:8000 root@dv-vdm"; pkill -f "127.0.0.1:18080:localhost"; exit 0; fi
S=${1:-dv-vdm}; LP=${2:-18000}; RP=${3:-18080}
LOG=${TMPDIR:-/tmp}/vdm_tunnel_$S.log
nohup bash "$HERE/tunnel_loop.sh" colab "$S" "$LP" >> "$LOG" 2>&1 < /dev/null &
nohup bash "$HERE/tunnel_loop.sh" gcp "$LP" "$RP" >> "$LOG" 2>&1 < /dev/null &
scp -q "$HERE/relay.py" gen-host:~/videoindex/vdm/relay.py
ssh gen-host "pkill -f '[r]elay.py 172.17.0.1 $RP'"   # separate call: the start command below would match itself
ssh gen-host "cd ~/videoindex/vdm && nohup python3 relay.py 172.17.0.1 $RP 127.0.0.1 $RP > relay_$RP.log 2>&1 < /dev/null &"
echo "tunnel started (log $LOG); endpoint: http://172.17.0.1:$RP"
