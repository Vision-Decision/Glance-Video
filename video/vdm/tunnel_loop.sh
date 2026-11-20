#!/usr/bin/env bash
# tunnel_loop.sh colab SESSION LP | gcp LP RP   -- one self-restarting ssh forward (used by tunnel.sh)
OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=15 -o ServerAliveCountMax=4
      -o ExitOnForwardFailure=yes -o LogLevel=ERROR)
while true; do
  if [ "$1" = colab ]; then
    ssh -N "${OPTS[@]}" -o "ProxyCommand=colab-pool ssh $2 --proxy-mode" -L "$3:localhost:8000" "root@$2"
  else
    ssh -N "${OPTS[@]}" -R "127.0.0.1:$3:localhost:$2" gen-host
  fi
  echo "$(date) $1 forward exited, restarting" ; sleep 5
done
