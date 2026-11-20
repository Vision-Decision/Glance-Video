# Deploy the frozen vdm v0 server on a Colab VM, plus a reverse tunnel and a keep-alive.
# All three run under setsid, so a kernel restart (or a lost exec connection) does not kill them.
#   python3 video/vdm/deploy_vm.py PORT > /tmp/d.py && colab-pool exec NAME --timeout 1500 -f /tmp/d.py
# This file is a generator: it embeds server.py verbatim (no logic changes) into a script run on the VM.
import sys
port = int(sys.argv[1])
src = open(__file__.replace("deploy_vm.py", "server.py")).read()
print(f'''
import subprocess, os, time, urllib.request
PORT = {port}
os.makedirs("/content/vdm", exist_ok=True)
open("/content/vdm/server.py", "w").write({src!r})
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True)
print(sh("pip install -q 'glance @ git+https://github.com/Vision-Decision/Glance' flash-linear-attention 2>&1 | tail -2").stdout, flush=True)
for _ in range(3):
    sh("python -m pip uninstall -y -q torchao; pip uninstall -y -q torchao")
print("torchao left:", sh("python -c 'import importlib.util as u;print(u.find_spec(\\"torchao\\"))'").stdout.strip(), flush=True)
open("/content/vdm/start.sh", "w").write("cd /content/vdm && setsid nohup python server.py --port 8000 --batch 32 --selftest > server.log 2>&1 &\\n")
# keep-alive / watchdog: every 5 min hit /health, restart the server if it is down
open("/content/vdm/keepalive.sh", "w").write("""#!/bin/bash
while true; do
  if ! curl -s -m 20 localhost:8000/health > /dev/null; then
    if ! pgrep -f "[s]erver.py --port 8000" > /dev/null; then echo "$(date) server down, restarting"; bash /content/vdm/start.sh; fi
  fi
  echo "$(date) alive" > /content/vdm/keepalive.last; sleep 300
done
""")
if not os.path.exists("/content/vdm/tun_key"):
    sh("ssh-keygen -q -t ed25519 -N '' -C vdm-colab-$(hostname) -f /content/vdm/tun_key")
# reverse tunnel with exponential backoff (5 s .. 120 s)
open("/content/vdm/rtunnel.sh", "w").write(f"""#!/bin/bash
b=5
while true; do
  start=$(date +%s)
  ssh -N -i /content/vdm/tun_key -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \\\\
      -o ExitOnForwardFailure=yes -R 127.0.0.1:{{PORT}}:localhost:8000 user@gen-host
  [ $(( $(date +%s) - start )) -gt 300 ] && b=5
  echo "$(date) rtunnel exited, retry in $b s"; sleep $b; b=$(( b < 120 ? b * 2 : 120 ))
done
""")
if sh("pgrep -f '[s]erver.py --port 8000'").returncode:
    sh("bash /content/vdm/start.sh")
# do not block the kernel while the model loads (a long exec loses its connection); poll /health from later execs
if sh("pgrep -f '[k]eepalive.sh'").returncode:
    sh("setsid nohup bash /content/vdm/keepalive.sh > /content/vdm/keepalive.log 2>&1 &")
if sh("pgrep -f '[r]tunnel.sh'").returncode:
    sh("setsid nohup bash /content/vdm/rtunnel.sh > /content/vdm/rtunnel.log 2>&1 &")
print("PUBKEY", open("/content/vdm/tun_key.pub").read().strip())
''')
