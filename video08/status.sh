#!/usr/bin/env bash
# our queue jobs: state + last meaningful log line
for n in $(grep -v '^#' ~/glance-queue/glance-video.tsv | cut -f1); do
  s=pending; for x in done failed pid; do [ -f ~/glance/queue/$n.$x ] && s=$x; done; [ $s = pid ] && s=running
  l=$(grep -v '\[transformers\]\|Warning\|warn(' ~/glance-queue/logs/$n.log 2>/dev/null | tail -1 | cut -c1-150)
  printf "%-36s %-8s %s\n" "${n#glance-video-}" $s "$l"
done
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' '; echo; date
