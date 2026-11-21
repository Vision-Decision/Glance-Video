#!/usr/bin/env bash
# queue entry point: q.sh MODULE ARGS...   (runs python -m video08.MODULE under ~/glance-video-exp/env.sh)
. $HOME/glance-video-exp/env.sh
cd $GV && exec $PY -m video08.$1 "${@:2}"
