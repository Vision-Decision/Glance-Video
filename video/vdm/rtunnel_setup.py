# Run on the Colab VM (colab-pool exec dv-vdm -f rtunnel_setup.py): direct reverse tunnel Colab:8000 -> gen-host
# 127.0.0.1:18080, no laptop and no colab-pool lock involved. One-time on gen-host, authorize the printed key with
#   restrict,port-forwarding,permitlisten="127.0.0.1:18080",command="/bin/true" <pubkey>
# in ~/.ssh/authorized_keys (port-forward only, one listen port, no shell). The relay on gen-host
# (relay.py 172.17.0.1 18080 127.0.0.1 18080) exposes it to containers on the default bridge.
import os, subprocess, time
GCP = os.environ.get("VDM_GCP", "user@gen-host")
if not os.path.exists("/content/vdm/tun_key"):
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "vdm-colab-dv-vdm", "-f", "/content/vdm/tun_key"])
print(open("/content/vdm/tun_key.pub").read().strip())
open("/content/vdm/rtunnel.sh", "w").write(f"""#!/bin/bash
while true; do
  ssh -N -i /content/vdm/tun_key -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \\
      -o ExitOnForwardFailure=yes -R 127.0.0.1:18080:localhost:8000 {GCP}
  echo "$(date) rtunnel exited, restarting"; sleep 5
done
""")
if subprocess.run(["pgrep", "-f", "rtunnel[.]sh"], capture_output=True).returncode:
    subprocess.run("nohup bash /content/vdm/rtunnel.sh > /content/vdm/rtunnel.log 2>&1 &", shell=True)
time.sleep(5)
print(open("/content/vdm/rtunnel.log").read()[-300:])
