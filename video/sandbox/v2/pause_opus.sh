#!/bin/bash
# let the in-flight Opus pair finish, then stop run.py before it launches more items
R=~/videoindex/data/runs
until [ -f $R/v2o_hard_base/claude-opus-5-5/S-EMBER_346/result.json ] && [ -f $R/v2o_hard_scan/claude-opus-5-5/S-EMBER_346/result.json ]; do sleep 5; done
for p in $(pgrep -f "[p]ython3 run.py --run v2o_hard"); do kill $p; done
sleep 2
for c in $(docker ps --format "{{.Names}}" | grep "^vi_v2o_hard_"); do n=${c#vi_v2o_hard_}; docker kill $c >/dev/null; arm=${n%%_*}; iid=${n#*_}; sudo rm -rf $R/v2o_hard_$arm/claude-opus-5-5/$iid; done
echo "opus paused $(date -u +%T)"
