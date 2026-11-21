#!/usr/bin/env bash
# writes ~/glance-queue/glance-video.tsv (the image experiments' queue picks it up after its own jobs).
# Concurrency is bounded by AFTER chains: lane T (S1 training), lane E (evals, one at a time, batched streams),
# plus the CPU jobs (decode cache, HF sync). M7 / M8 / throughput run on the Colab G4 (run_g4.sh).
GV=$HOME/glance-video-exp; Q=$GV/code/video08/q.sh; O=$GV/out; V=$GV/data/raw/charades/Charades_v1_480
j() { printf "%s\t%s\t%s\t%s\t%s\n" "glance-video-$1" any "$2" "$3" "$4"; }
chain() { local prev=$1; shift; while [ $# -gt 0 ]; do j "$1" "$2" "$prev" "$3"; prev=glance-video-$1; shift 3; done; }
{
echo "# glance-video 0.8B (owner: video agent). NAME GPU MEM_GB AFTER COMMAND; logs in ~/glance-queue/logs"
j smoke 6 - "$Q tests.test_stream cuda"
j upload 0 - "$GV/code/video08/upload_loop.sh"
j cache256 0 - "$Q data.cache $GV/data 8"
j prep-rep 0 - "$GV/code/video08/data/prep_rep.sh"
j test-batch 6 - "$Q tests.test_batch $V/001YG.mp4 $V/004QE.mp4 $V/003WS.mp4"
# already running when the lanes were introduced (kept)
j m6-presence-stream-fps2 6 glance-video-smoke "$Q evals.m6 --bench presence --arm stream --fps 2 --n 300 --out $O/m6/presence_stream_fps2.jsonl"
j m6-pushup-stream-fps2 6 glance-video-smoke "$Q evals.m6 --bench pushup --arm stream --fps 2 --n 215 --out $O/m6/pushup_stream_fps2.jsonl"
j deploy-stream-fps2 8 glance-video-smoke "$Q evals.deploy --split eval --fps 2 --arm stream --streams 12 --minutes 10 --rules 16 --out $O/deploy"
# lane T
chain glance-video-prep-rep \
  s1-smoke 10 "$Q train_s1 --out $O/p3/smoke --steps 20 --save 60 --workers 3 --batch 4" \
  s1a 24 "$Q train_s1 --out $O/p3/s1a --steps 1500 --workers 6" \
  s1b-lora16 26 "$Q train_s1 --out $O/p3/s1b_lora16 --steps 1500 --lora 16 --workers 6"
# lane E (batched m6: GV_BATCH=8 lockstep streams per forward)
chain glance-video-m6-presence-stream-fps2,glance-video-test-batch \
  m6-presence-stream-fps30 16 "$Q evals.m6 --bench presence --arm stream --fps 30 --n 300 --out $O/m6/presence_stream_fps30.jsonl" \
  m6-sta-stream-fps2 12 "$Q evals.m6 --bench sta --arm stream --fps 2 --n 300 --out $O/m6/sta_stream_fps2.jsonl" \
  m6-sta-stream-fps30 16 "$Q evals.m6 --bench sta --arm stream --fps 30 --n 300 --out $O/m6/sta_stream_fps30.jsonl" \
  m6-pushup-stream-fps30 16 "$Q evals.m6 --bench pushup --arm stream --fps 30 --n 215 --out $O/m6/pushup_stream_fps30.jsonl" \
  deploy-stream-fps30 8 "$Q evals.deploy --split eval --fps 30 --arm stream --streams 12 --minutes 10 --rules 16 --out $O/deploy" \
  deploy-recent4-fps2 6 "$Q evals.deploy --split eval --fps 2 --arm recent4 --streams 12 --minutes 10 --rules 16 --layers '' --out $O/deploy" \
  probe-train-fps2 8 "$Q evals.deploy --split train --fps 2 --arm none --streams 40 --minutes 10 --out $O/deploy_train" \
  probe-train-fps30 8 "$Q evals.deploy --split train --fps 30 --arm none --streams 16 --minutes 10 --out $O/deploy_train" \
  m6-presence-native-fps2 16 "$Q evals.m6 --bench presence --arm native --fps 2 --n 300 --out $O/m6/presence_native_fps2.jsonl" \
  m6-sta-native-fps2 16 "$Q evals.m6 --bench sta --arm native --fps 2 --n 300 --out $O/m6/sta_native_fps2.jsonl" \
  m6-pushup-native-fps2 24 "$Q evals.m6 --bench pushup --arm native --fps 2 --n 215 --out $O/m6/pushup_native_fps2.jsonl" \
  m6-pushup-native-fps30 24 "$Q evals.m6 --bench pushup --arm native --fps 30 --n 215 --out $O/m6/pushup_native_fps30.jsonl"
} > $HOME/glance-queue/glance-video.tsv.tmp && mv $HOME/glance-queue/glance-video.tsv.tmp $HOME/glance-queue/glance-video.tsv
