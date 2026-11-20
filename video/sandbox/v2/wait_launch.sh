#!/bin/bash
# wait_launch.sh RUN IDS : start run_v2.sh RUN --ids IDS once every listed item has a transcript
RUN=$1; IDS=$2
cd ~/videoindex
while true; do
  miss=$(python3 -c "
import json,os,sys
ids=set(sys.argv[1].split(\",\"))
for l in open(\"research/data/test.jsonl\"):
    x=json.loads(l)
    if x[\"item_id\"] in ids and not os.path.exists(\"data/transcripts/\"+os.path.basename(x[\"video\"])+\".tsv\"): print(x[\"item_id\"])" "$IDS")
  [ -z "$miss" ] && break
  sleep 30
done
cd v2 && CONC=${CONC:-4} exec ./run_v2.sh $RUN --ids $IDS
