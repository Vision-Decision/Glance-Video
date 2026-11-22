#!/usr/bin/env bash
# G4 job chain: runs /content/gv/chain/todo/*.sh in name order, one at a time. A failing job stops the chain
# (chain/STATE = failed:<job>); an empty chain sets STATE = idle, and the Mac watcher (g4_watch.sh) then syncs the
# results and shuts the VM down. New work = drop a script into chain/todo/.
C=/content/gv/chain; mkdir -p $C/todo $C/done $C/failed; cd /content/gv
while true; do
  j=$(ls $C/todo/*.sh 2>/dev/null | head -1)
  if [ -z "$j" ]; then echo idle > $C/STATE; sleep 60; continue; fi
  n=$(basename $j); echo "running:$n" > $C/STATE; echo "$(date +%H:%M:%S) start $n" >> logs/progress.log
  if bash $j > logs/chain_$n.log 2>&1; then mv $j $C/done/; echo "$(date +%H:%M:%S) done $n" >> logs/progress.log
  else mv $j $C/failed/; echo "failed:$n" > $C/STATE; echo "$(date +%H:%M:%S) FAILED $n" >> logs/progress.log; exit 1; fi
done
