#!/usr/bin/env bash
# writes ~/glance-queue/glance-video.tsv (the image experiments' queue picks it up after its own jobs). Order = priority.
GV=$HOME/glance-video-exp; Q=$GV/code/video08/q.sh; O=$GV/out
j() { printf "%s\t%s\t%s\t%s\t%s\n" "glance-video-$1" any "$2" "$3" "$4"; }
{
echo "# glance-video 0.8B (owner: video agent). NAME GPU MEM_GB AFTER COMMAND; logs in ~/glance-queue/logs"
j smoke 6 - "$Q tests.test_stream cuda"
# --- P3 smoke (short) then the two S1 arms
j upload 0 - "$GV/code/video08/upload_loop.sh"
j prep-rep 0 - "$GV/code/video08/data/prep_rep.sh"
j s1-smoke 10 glance-video-prep-rep "$Q train_s1 --out $O/p3/smoke --steps 30 --save 60 --workers 2"
j s1a 14 glance-video-s1-smoke "$Q train_s1 --out $O/p3/s1a --steps 4000"
# --- P2
for b in presence sta; do for f in 2 30; do
  j m6-$b-stream-fps$f 6 glance-video-smoke "$Q evals.m6 --bench $b --arm stream --fps $f --n 300 --out $O/m6/${b}_stream_fps$f.jsonl"; done; done
for f in 2 30; do j m6-pushup-stream-fps$f 6 glance-video-smoke "$Q evals.m6 --bench pushup --arm stream --fps $f --n 215 --out $O/m6/pushup_stream_fps$f.jsonl"; done
# M7 / M8 / throughput run on the Colab G4 (video08/run_g4.sh), not here
for f in 2 30; do j deploy-stream-fps$f 8 glance-video-smoke "$Q evals.deploy --split eval --fps $f --arm stream --streams 12 --minutes 10 --rules 16 --out $O/deploy"; done
j deploy-recent4-fps2 6 glance-video-smoke "$Q evals.deploy --split eval --fps 2 --arm recent4 --streams 12 --minutes 10 --rules 16 --layers '' --out $O/deploy"
j probe-train-fps2 8 glance-video-smoke "$Q evals.deploy --split train --fps 2 --arm none --streams 40 --minutes 10 --out $O/deploy_train"
j probe-train-fps30 8 glance-video-smoke "$Q evals.deploy --split train --fps 30 --arm none --streams 16 --minutes 10 --out $O/deploy_train"
j s1b-lora16 16 glance-video-s1-smoke "$Q train_s1 --out $O/p3/s1b_lora16 --steps 4000 --lora 16"
j m6-presence-native-fps2 16 glance-video-smoke "$Q evals.m6 --bench presence --arm native --fps 2 --n 300 --out $O/m6/presence_native_fps2.jsonl"
j m6-sta-native-fps2 16 glance-video-smoke "$Q evals.m6 --bench sta --arm native --fps 2 --n 300 --out $O/m6/sta_native_fps2.jsonl"
for f in 2 30; do j m6-pushup-native-fps$f 24 glance-video-smoke "$Q evals.m6 --bench pushup --arm native --fps $f --n 215 --out $O/m6/pushup_native_fps$f.jsonl"; done
} > $HOME/glance-queue/glance-video.tsv.tmp && mv $HOME/glance-queue/glance-video.tsv.tmp $HOME/glance-queue/glance-video.tsv
