#!/bin/bash
# Keep the scan VM off while the Opus run waits for its 5-hour subscription window (no idle GPU), and back on in time.
#   scan_vm_cycle.sh [NAME] [LOG]     (operator machine; runs until the Opus run ends)
# Watches run.py's log on gen-host. On a new "5-hour window at X: waiting N min" line with N > MIN_WAIT, it waits until
# no Opus agent container is running, stops the VM, and restores it (scan_vm.sh restore) LEAD minutes before the wait
# ends. When the run is finished or aborted it stops the VM and exits. Events go to stdout with UTC timestamps.
set -uo pipefail
NAME=${1:-dv-scan}; LOG=${2:-~/videoindex/data/v2o_8.log}
MIN_WAIT=${MIN_WAIT:-45}; LEAD=${LEAD:-30}; GCP=${GCP:-gen-host}
HERE=$(cd "$(dirname "$0")" && pwd)
say() { echo "$(date -u +%FT%T) $*"; }
seen=$(ssh "$GCP" "grep -c 'waiting .* min for its reset' $LOG" 2>/dev/null || echo 0)
up=1
say "watching $LOG (waits seen so far: $seen)"
while true; do
  st=$(ssh "$GCP" "grep -c 'waiting .* min for its reset' $LOG; tail -n 3 $LOG | grep -c -E '^done:|^ABORT'; \
                   pgrep -f '[p]ython3 run.py --run v2o_hard' | wc -l; grep 'waiting .* min for its reset' $LOG | tail -1" 2>/dev/null) || { sleep 60; continue; }
  n=$(sed -n 1p <<<"$st"); fin=$(sed -n 2p <<<"$st"); alive=$(sed -n 3p <<<"$st"); last=$(sed -n 4p <<<"$st")
  if [ "${fin:-0}" -gt 0 ] || [ "${alive:-1}" = 0 ]; then
    [ $up = 1 ] && "$HERE/scan_vm.sh" stop "$NAME"
    say "Opus run finished or stopped; VM stopped; exiting"; exit 0
  fi
  if [ "${n:-0}" -gt "$seen" ]; then
    seen=$n
    mins=$(sed -E 's/.*waiting ([0-9]+) min.*/\1/' <<<"$last")
    t_wait=$(date +%s)
    if [ "$mins" -gt "$MIN_WAIT" ] && [ $up = 1 ]; then
      until [ "$(ssh "$GCP" "docker ps --format '{{.Names}}' | grep -c '^vi_v2o_'" 2>/dev/null)" = 0 ]; do sleep 30; done
      "$HERE/scan_vm.sh" stop "$NAME" && up=0 && say "stopped $NAME for a ${mins}-min wait"
      wake=$(( t_wait + (mins - LEAD) * 60 ))
      while [ "$(date +%s)" -lt "$wake" ]; do sleep 60; done
      say "restoring $NAME"
      for try in 1 2; do
        if "$HERE/scan_vm.sh" restore "$NAME"; then up=1; break; fi
        say "restore attempt $try failed"; "$HERE/scan_vm.sh" stop "$NAME" || true
      done
      [ $up = 1 ] || { say "RESTORE FAILED: scan VM is down"; exit 1; }
      seen=$(ssh "$GCP" "grep -c 'waiting .* min for its reset' $LOG")
    fi
  fi
  sleep 60
done
