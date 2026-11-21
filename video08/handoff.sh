#!/bin/zsh
# 07:41 CST: A100 window closes -> move S1a/S1b to the G4 (checkpoint copy via the Mac; ~140 MB each)
D=/private/tmp/claude-501/-Users-aac6fef-Developer-decision-vison/3d07ac00-1b48-49c7-9f7e-e96c122b52d4/scratchpad/g4
while [ $(TZ=Asia/Shanghai date +%H%M) -lt 0741 ] || [ $(TZ=Asia/Shanghai date +%H%M) -gt 1200 ]; do sleep 60; done
for i in $(seq 1 20); do ssh a100 'ls ~/glance/queue/glance-video-s1a.pid ~/glance/queue/glance-video-s1b-lora16.pid 2>/dev/null' | grep -q pid || break; sleep 30; done
for r in s1a s1b_lora16; do mkdir -p $D/ck/$r; scp -q a100:glance-video-exp/out/p3/$r/'{ckpt_last.pt,log.jsonl}' $D/ck/$r/ 2>/dev/null || scp -q a100:glance-video-exp/out/p3/$r/ckpt_last.pt a100:glance-video-exp/out/p3/$r/log.jsonl $D/ck/$r/; scp -q a100:glance-video-exp/out/p3/$r/ckpt_final.pt $D/ck/$r/ 2>/dev/null; done
ls -la $D/ck/*
colab-pool exec glance-video-g4 -f $D/sh.py --env "CMD=mkdir -p /content/gv/out/p3/s1a /content/gv/out/p3/s1b_lora16"
for r in s1a s1b_lora16; do for f in $D/ck/$r/*; do colab-pool upload glance-video-g4 $f /content/gv/out/p3/$r/$(basename $f); done; done
colab-pool upload glance-video-g4 $D/g4_day.sh /content/gv/g4_day.sh
colab-pool exec glance-video-g4 -f $D/sh.py --env "CMD=cd /content/gv && (setsid nohup bash g4_day.sh > logs_day.txt 2>&1 &); sleep 2; tail -2 logs/progress.log"
# the A100 queue must not resume these tonight (G4 owns them now)
ssh a100 'for n in s1a s1b-lora16 p3-s1a-fps2 p3-s1a-fps30 p3-s1b-fps2 p3-s1b-fps30; do echo moved-to-g4 > ~/glance/queue/glance-video-$n.done; done'
echo "handoff done $(date)"
