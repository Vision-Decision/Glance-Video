#!/usr/bin/env bash
# background CPU/disk job: HC-STVG v2 videos from ModelScope (zfxzfx/hc-stvg2, 10 zips, ~115 GB). Parked until S1 decides.
# Stops if the shared disk would drop below 150 GB free.
O=$HOME/glance-video-exp/data/raw/hcstvg2; mkdir -p $O; cd $O
for f in 0.zip 1.zip "2 (1).zip" 3.zip 4.zip 5.zip 6.zip 7.zip 8.zip 9.zip; do
  [ -f "$f" ] && continue
  free=$(df --output=avail -BG $HOME | tail -1 | tr -dc 0-9)
  [ "$free" -lt 165 ] && { echo "disk guard: ${free}G free, stop before $f"; exit 0; }
  u="https://www.modelscope.cn/api/v1/datasets/zfxzfx/hc-stvg2/repo?Revision=master&FilePath=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))" "$f")"
  nice -n 15 $HOME/glance/venv/bin/python $HOME/glance-video-exp/code/video08/data/pdl.py "$u" "$O/$f" 8 && echo "got $f"
done
echo all-done
