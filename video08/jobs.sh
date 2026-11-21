#!/usr/bin/env bash
# writes ~/glance-queue/glance-video.tsv for the training agent's queue. Order = priority; AFTER chains bound concurrency:
#   CPU: decode cache (10 procs) -> train-part cache (10 procs); S1 training uses 3 decode workers per run -> <= 16 decoders
#   evals: two lanes (E1, E2), both start only after the eval decode cache (mp4 256 + fps-2 npy) is complete
# M7 / M8 / throughput run on the Colab G4 (run_g4.sh).
GV=$HOME/glance-video-exp; Q=$GV/code/video08/q.sh; O=$GV/out; V=$GV/data/raw/charades/Charades_v1_480
j() { printf "%s\t%s\t%s\t%s\t%s\n" "glance-video-$1" any "$2" "$3" "$4"; }
chain() { local prev=$1; shift; while [ $# -gt 0 ]; do j "$1" "$2" "$prev" "$3"; prev=glance-video-$1; shift 3; done; }
C=glance-video-cache256
{
echo "# glance-video 0.8B (owner: video agent). NAME GPU MEM_GB AFTER COMMAND; logs in ~/glance-queue/logs"
j s1a 12 - "$Q train_s1 --out $O/p3/s1a --steps 1600 --batch 4 --qbatch 4 --workers 6"
j s1b-lora16 14 - "$Q train_s1 --out $O/p3/s1b_lora16 --steps 1600 --batch 4 --qbatch 4 --lora 16 --workers 6"
j upload 0 - "$GV/code/video08/upload_loop.sh"
j cache256 0 - "$Q data.cache $GV/data 10 eval"
j cache256-train 0 $C "$Q data.cache $GV/data 24 train"
# lane T: smoke, then both S1 arms in parallel
# s1-smoke: 16/20 steps passed (losses fall, all four heads train); not re-queued
# j s1-smoke 24 - "$Q train_s1 --out $O/p3/smoke --steps 20 --save 60 --workers 3 --batch 4"
# lane E1 (batched M6)
chain $C \
  m6-presence-stream-fps2 12 "$Q evals.m6 --bench presence --arm stream --fps 2 --n 300 --out $O/m6/presence_stream_fps2.jsonl" \
  m6-presence-stream-fps30 16 "$Q evals.m6 --bench presence --arm stream --fps 30 --n 300 --out $O/m6/presence_stream_fps30.jsonl" \
  m6-sta-stream-fps2 12 "$Q evals.m6 --bench sta --arm stream --fps 2 --n 300 --out $O/m6/sta_stream_fps2.jsonl" \
  m6-sta-stream-fps30 16 "$Q evals.m6 --bench sta --arm stream --fps 30 --n 300 --out $O/m6/sta_stream_fps30.jsonl" \
  m6-pushup-stream-fps30 16 "$Q evals.m6 --bench pushup --arm stream --fps 30 --n 215 --out $O/m6/pushup_stream_fps30.jsonl" \
  m6-presence-native-fps2 16 "$Q evals.m6 --bench presence --arm native --fps 2 --n 300 --out $O/m6/presence_native_fps2.jsonl" \
  m6-sta-native-fps2 16 "$Q evals.m6 --bench sta --arm native --fps 2 --n 300 --out $O/m6/sta_native_fps2.jsonl" \
  m6-pushup-native-fps2 24 "$Q evals.m6 --bench pushup --arm native --fps 2 --n 215 --out $O/m6/pushup_native_fps2.jsonl" \
  m6-pushup-native-fps30 24 "$Q evals.m6 --bench pushup --arm native --fps 30 --n 215 --out $O/m6/pushup_native_fps30.jsonl"
# lane E3: R1 zero-shot robustness grid (fps x tokens per group x aspect ratio); presence n=150, PushupBench n=100
chain $C \
  r1-pres-f1-256 12 "$Q evals.m6 --bench presence --arm stream --fps 1 --n 150 --out $O/r1/presence_f1_256.jsonl" \
  r1-pres-f5-256 12 "$Q evals.m6 --bench presence --arm stream --fps 5 --n 150 --out $O/r1/presence_f5_256.jsonl" \
  r1-pres-f10-256 14 "$Q evals.m6 --bench presence --arm stream --fps 10 --n 150 --out $O/r1/presence_f10_256.jsonl" \
  r1-pres-f2-128 10 "$Q evals.m6 --bench presence --arm stream --fps 2 --size 128 --n 150 --out $O/r1/presence_f2_128.jsonl" \
  r1-pres-f2-512 16 "$Q evals.m6 --bench presence --arm stream --fps 2 --size 512 --n 150 --out $O/r1/presence_f2_512.jsonl" \
  r1-pres-f30-128 14 "$Q evals.m6 --bench presence --arm stream --fps 30 --size 128 --n 150 --out $O/r1/presence_f30_128.jsonl" \
  r1-pres-f30-512 24 "$Q evals.m6 --bench presence --arm stream --fps 30 --size 512 --n 150 --out $O/r1/presence_f30_512.jsonl" \
  r1-pres-f2-352x192 12 "$Q evals.m6 --bench presence --arm stream --fps 2 --size 352x192 --n 150 --out $O/r1/presence_f2_352x192.jsonl" \
  r1-pres-f2-192x352 12 "$Q evals.m6 --bench presence --arm stream --fps 2 --size 192x352 --n 150 --out $O/r1/presence_f2_192x352.jsonl" \
  r1-push-f1-256 12 "$Q evals.m6 --bench pushup --arm stream --fps 1 --n 100 --out $O/r1/pushup_f1_256.jsonl" \
  r1-push-f5-256 12 "$Q evals.m6 --bench pushup --arm stream --fps 5 --n 100 --out $O/r1/pushup_f5_256.jsonl" \
  r1-push-f10-256 14 "$Q evals.m6 --bench pushup --arm stream --fps 10 --n 100 --out $O/r1/pushup_f10_256.jsonl" \
  r1-push-f2-512 16 "$Q evals.m6 --bench pushup --arm stream --fps 2 --size 512 --n 100 --out $O/r1/pushup_f2_512.jsonl" \
  r1-push-f2-352x192 12 "$Q evals.m6 --bench pushup --arm stream --fps 2 --size 352x192 --n 100 --out $O/r1/pushup_f2_352x192.jsonl"
# lane E2 (deployment protocol, probe features, then P3 eval once S1a is final)
chain $C \
  deploy-stream-fps2 8 "$Q evals.deploy --split eval --fps 2 --arm stream --streams 12 --minutes 10 --rules 16 --out $O/deploy" \
  deploy-stream-fps30 10 "$Q evals.deploy --split eval --fps 30 --arm stream --streams 12 --minutes 10 --rules 16 --out $O/deploy" \
  deploy-recent4-fps2 6 "$Q evals.deploy --split eval --fps 2 --arm recent4 --streams 12 --minutes 10 --rules 16 --layers '' --out $O/deploy" \
  probe-train-fps2 8 "$Q evals.deploy --split train --fps 2 --arm none --streams 40 --minutes 10 --out $O/deploy_train" \
  probe-train-fps30 10 "$Q evals.deploy --split train --fps 30 --arm none --streams 16 --minutes 10 --out $O/deploy_train"
chain glance-video-s1a \
  p3-s1a-fps2 16 "$Q evals.p3_eval --ckpt $O/p3/s1a/ckpt_final.pt --fps 2 --n 300 --out $O/p3/eval_s1a_fps2.jsonl" \
  p3-s1a-fps30 20 "$Q evals.p3_eval --ckpt $O/p3/s1a/ckpt_final.pt --fps 30 --n 300 --out $O/p3/eval_s1a_fps30.jsonl"
chain glance-video-s1b-lora16 \
  p3-s1b-fps2 16 "$Q evals.p3_eval --ckpt $O/p3/s1b_lora16/ckpt_final.pt --fps 2 --n 300 --out $O/p3/eval_s1b_fps2.jsonl" \
  p3-s1b-fps30 20 "$Q evals.p3_eval --ckpt $O/p3/s1b_lora16/ckpt_final.pt --fps 30 --n 300 --out $O/p3/eval_s1b_fps30.jsonl"
} > $HOME/glance-queue/glance-video.tsv.tmp && mv $HOME/glance-queue/glance-video.tsv.tmp $HOME/glance-queue/glance-video.tsv
