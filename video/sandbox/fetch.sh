#!/usr/bin/env bash
# Robust download of what the sandbox needs (resumable plain HTTP; xet kept failing on this network).
#   bash video/sandbox/fetch.sh            # whisper models + all v1.1 videos into $VI_DATA (default data/video_index/)
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
D="${VI_DATA:-$ROOT/data/video_index}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" HF_HUB_DOWNLOAD_TIMEOUT=120
retry() { for i in $(seq 1 50); do "$@" && return 0; echo "retry $i: $*" >&2; sleep 15; done; return 1; }
mkdir -p "$D/models/faster-whisper"
retry hf download Systran/faster-whisper-large-v3 --local-dir "$D/models/faster-whisper/large-v3" && echo WHISPER_LARGE_OK
retry hf download Systran/faster-whisper-small --local-dir "$D/models/faster-whisper/small" && echo WHISPER_SMALL_OK
retry hf download Video-Index/Video-Index-v1-1 --repo-type dataset --include "videos/*" --include "items/*" \
      --local-dir "$D/v1_1" && echo VIDEOS_OK
