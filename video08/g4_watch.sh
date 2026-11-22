#!/bin/zsh
# Mac watcher for the G4 chain: every 5 min; when the chain is idle (or failed) and no video08 python runs on the VM,
# pull out/ + logs, push them to HF (video08/g4), then stop the VM. No human needed.
D=/private/tmp/claude-501/-Users-aac6fef-Developer-decision-vison/3d07ac00-1b48-49c7-9f7e-e96c122b52d4/scratchpad/g4
while true; do
  st=$(timeout 100 colab-pool exec glance-video-g4 -f $D/sh.py --timeout 90 --env "CMD=cat /content/gv/chain/STATE 2>/dev/null; pgrep -fc 'python.*video08' || true" 2>/dev/null | grep -v -E "Switched|LIBARCHIVE" | tr '\n' ' ')
  echo "$(date +%H:%M) $st"
  if echo "$st" | grep -qE "^(idle|failed:[^ ]*) +0 *$"; then
    timeout 200 colab-pool exec glance-video-g4 -f $D/sh.py --timeout 190 --env "CMD=cd /content/gv && tar czf /content/res.tgz out/m7 out/m8 out/r1 out/throughput out/p3/*.jsonl out/p3/*/log.jsonl out/p3/*/config.json out/p3/*/ckpt_final.pt out/p3/*/ckpt_step*.pt logs chain 2>/dev/null; ls -la /content/res.tgz" >/dev/null 2>&1
    if timeout 900 colab-pool download glance-video-g4 /content/res.tgz $D/res/final.tgz >/dev/null 2>&1; then
      rm -rf $D/res/final && mkdir -p $D/res/final && tar xzf $D/res/final.tgz -C $D/res/final && cp $D/res/README.md $D/res/final/ 2>/dev/null
      (cd /tmp && uvx --from huggingface_hub python -c "
from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='$D/res/final', path_in_repo='video08/g4', repo_id='Vision-Decision/glance-train-ckpts', commit_message='video08/g4: chain finished, final sync before shutdown')
print('ok')" 2>&1 | tail -1)
      echo "$(date +%H:%M) final sync done; stopping VM ($st)"; colab-pool stop glance-video-g4; rm -f $D/res/final.tgz; exit 0
    fi
  fi
  sleep 300
done
