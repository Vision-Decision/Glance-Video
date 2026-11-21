#!/usr/bin/env bash
# CPU job: wait for the RepCount-A train lance to finish downloading, then extract mp4s + index
. $HOME/glance-video-exp/env.sh
F=$GV/data/raw/repcount/data/train.lance/data/1100111101110011110001104e723e432eb7fd7d6e774d6238.lance
until [ -f "$F" ]; do sleep 30; done
$PY $GV/code/video08/data/repcount.py $GV/data train
