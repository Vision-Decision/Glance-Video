#!/usr/bin/env bash
# queue entry point: q.sh MODULE ARGS...   (runs python -m video08.MODULE under ~/glance-video-exp/env.sh)
. $HOME/glance-video-exp/env.sh
export TORCHINDUCTOR_COMPILE_THREADS=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=4
cd $GV && exec $PY -m video08.$1 "${@:2}"
