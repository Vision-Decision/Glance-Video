#!/usr/bin/env bash
# Detached launcher on the run host:  bash sandbox/launch.sh LOGNAME [env VAR=.. ...] -- run.py args...
# e.g. bash sandbox/launch.sh full1_a -- --run full1 --exclude-run pilot10 --concurrency 8 --cpus 1.25
cd "$(dirname "$0")/.." || exit 1
LOG="$1"; shift
ENVS=()
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do ENVS+=("$1"); shift; done
shift
export VI_DATA="$PWD/data" VI_RESEARCH="$PWD/research" VI_VDM="$PWD/vdmclient"
setsid nohup env "${ENVS[@]}" python3 sandbox/run.py "$@" > "data/$LOG.log" 2>&1 < /dev/null &
echo "started $LOG pid $!"
