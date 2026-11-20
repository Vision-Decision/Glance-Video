#!/bin/bash
# start the DeepSeek paired hard run at the off-peak boundary (10:00 UTC)
target=$(date -u -d "10:00:30" +%s); now=$(date -u +%s); [ $target -gt $now ] && sleep $((target-now))
cd ~/videoindex/v2 && CONC=3 exec ./run_v2.sh v2_hard --set hard60
