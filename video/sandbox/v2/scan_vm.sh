#!/bin/bash
# Stop / restore the Glance-Video scan VM (scan server + GPU whisper) behind the reverse tunnel to gen-host.
#   scan_vm.sh restore NAME [SCAN_PORT] [WHISPER_PORT]   new G4 -> deps + ckpt (HF) -> fresh tunnel key -> servers up
#   scan_vm.sh stop NAME                                  stop the VM (state lives in RESTORE_DIR + HF, nothing to save)
# Runs on the operator machine (colab-pool). RESTORE_DIR holds the code tarball, CALIB.json and the loop/server scripts
# (pulled from the live VM once; backup on gen-host ~/videoindex/v2/restore). A new ed25519 key is generated on every
# VM and authorised on gen-host for localhost-only port forwarding (marker "vi-scan-tunnel <NAME>"); no private key ever
# leaves the VM. Prints "RESTORED <NAME> in <s> s" when /health answers through the tunnel.
set -euo pipefail
ACT=${1:?restore|stop}; NAME=${2:?name}; SPORT=${3:-18084}; WPORT=${4:-18090}
RESTORE_DIR=${RESTORE_DIR:-$(dirname "$0")/../../../.scan_vm_restore}
GCP=${GCP:-gen-host}; GCP_ADDR=${GCP_ADDR:-user@gen-host}
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT

if [ "$ACT" = stop ]; then
  colab-pool stop "$NAME"
  ssh "$GCP" "sed -i '/vi-scan-tunnel $NAME\$/d' ~/.ssh/authorized_keys"
  echo "STOPPED $NAME"; exit 0
fi

t0=$(date +%s)
colab-pool new --if-missing "$NAME" --gpu g4 ${VM_ACCOUNT:+--account "$VM_ACCOUNT" --allow-other-account} >/dev/null
for f in dv_v1.tgz:dv.tgz CALIB.json:CALIB.json vm_setup_scan.sh:vm_setup_scan.sh serve_loop.sh:serve_loop.sh \
         whisper_server.py:whisper_server.py ws_loop.sh:ws_loop.sh; do
  colab-pool upload "$NAME" "$RESTORE_DIR/${f%%:*}" "/content/${f##*:}" >/dev/null
done
install -m 600 ~/.cache/huggingface/token "$T/hftok"
colab-pool upload "$NAME" "$T/hftok" /content/hftok >/dev/null
cat > "$T/boot.py" <<EOF
import subprocess
r = subprocess.run("bash /content/vm_setup_scan.sh > /content/setup.log 2>&1; tail -1 /content/setup.log", shell=True, capture_output=True, text=True)
print(r.stdout.strip())
subprocess.run("mkdir -p /content/vdm && rm -f /content/vdm/tun_key* && ssh-keygen -q -t ed25519 -N '' -C vi-scan-tunnel -f /content/vdm/tun_key", shell=True)
for port, local, fn in (($SPORT, 8000, "rtunnel.sh"), ($WPORT, 8090, "rtunnel_ws.sh")):
    open("/content/vdm/" + fn, "w").write(f"""#!/bin/bash
while true; do
  ssh -N -i /content/vdm/tun_key -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \\\\
      -o ExitOnForwardFailure=yes -R 127.0.0.1:{port}:localhost:{local} $GCP_ADDR
  sleep 5
done
""")
print("PUBKEY " + open("/content/vdm/tun_key.pub").read().split()[1])
EOF
out=$(colab-pool exec "$NAME" -f "$T/boot.py" --timeout 1800)
echo "$out" | grep -q "DONE" || { echo "setup failed: $(echo "$out" | tail -3)"; exit 1; }
pub=$(echo "$out" | sed -n 's/^PUBKEY //p')
ssh "$GCP" "sed -i '/vi-scan-tunnel $NAME\$/d' ~/.ssh/authorized_keys; echo 'restrict,port-forwarding,permitlisten=\"127.0.0.1:$SPORT\",permitlisten=\"127.0.0.1:$WPORT\",command=\"/bin/true\" ssh-ed25519 $pub vi-scan-tunnel $NAME' >> ~/.ssh/authorized_keys"
cat > "$T/start.py" <<'EOF'
import subprocess
for s in ("/content/serve_loop.sh", "/content/ws_loop.sh", "/content/vdm/rtunnel.sh", "/content/vdm/rtunnel_ws.sh"):
    subprocess.Popen(["setsid", "nohup", "bash", s], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True, cwd="/content")
print("started")
EOF
colab-pool exec "$NAME" -f "$T/start.py" >/dev/null
for i in $(seq 1 120); do
  if ssh "$GCP" "curl -s -m 5 127.0.0.1:$SPORT/health | grep -q scan-v1 && curl -s -m 5 127.0.0.1:$WPORT/ | grep -q whisper"; then
    echo "RESTORED $NAME in $(( $(date +%s) - t0 )) s"; exit 0
  fi
  sleep 10
done
echo "restore of $NAME did not come up within 20 min after setup"; exit 1
