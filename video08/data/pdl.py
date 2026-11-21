"""Parallel HTTP range download with resume:  python video08/data/pdl.py URL OUT [connections]"""
import os, sys, threading, time, urllib.request

url, out = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 16
size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60).headers["Content-Length"])
piece = 64 << 20; parts = [(i, min(i + piece, size) - 1) for i in range(0, size, piece)]
pd = out + ".parts"; os.makedirs(pd, exist_ok=True); lock = threading.Lock(); todo = list(parts)


def work():
    while True:
        with lock:
            if not todo:
                return
            a, b = todo.pop(0)
        fn = f"{pd}/{a:015d}"
        for k in range(200):
            have = os.path.getsize(fn) if os.path.exists(fn) else 0
            if have >= b - a + 1:
                break
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={a + have}-{b}"})
                with urllib.request.urlopen(req, timeout=15) as r, open(fn, "ab") as f:
                    t0, got = time.time(), 0
                    while True:
                        x = r.read(1 << 20)
                        if not x:
                            break
                        f.write(x); got += len(x)
                        if time.time() - t0 > 10:  # a throttled connection: reconnect (fresh ones are fast)
                            if got < (2 << 20):
                                break
                            t0, got = time.time(), 0
            except Exception as e:
                time.sleep(3)


ts = [threading.Thread(target=work) for _ in range(n)]
[t.start() for t in ts]; [t.join() for t in ts]
with open(out + ".tmp", "wb") as f:
    for a, b in parts:
        fn = f"{pd}/{a:015d}"; assert os.path.getsize(fn) == b - a + 1, fn
        f.write(open(fn, "rb").read())
os.replace(out + ".tmp", out)
import shutil; shutil.rmtree(pd)
print("ok", out, size)
