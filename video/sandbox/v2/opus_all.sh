#!/bin/bash
# Opus paired: the 8-item stratified subset first, then the rest of the 64 (resumable; finished items are skipped).
# run.py throttles on the subscription: waits for the 5-hour reset above 85%, stops above 70% of the 7-day window.
cd ~/videoindex/v2
./run_v2o.sh v2o_hard --ids S-EMBER_346,S-EMBER_3231,S-EMBER_2431,S-EMBER_7308,UrbanVideo-Bench_5348,ALLVB_20839,UrbanVideo-Bench_962,ALLVB_12953
grep -q "^ABORT" ~/videoindex/data/v2o_8.log && exit 0
echo "=== subset done, full 64 ===" 
exec ./run_v2o.sh v2o_hard --set hard60
