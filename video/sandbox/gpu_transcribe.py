#!/usr/bin/env python3
"""Transcripts for the VI videos via the GPU whisper-small service on the scan VM (reverse tunnel 127.0.0.1:18090).
Audio is extracted here (16 kHz mono Opus), sent over our own tunnel, transcribed in memory, nothing is stored there.
  python3 gpu_transcribe.py OUTDIR [--set hard60] [-j 3]"""
import argparse, concurrent.futures as cf, json, os, subprocess, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as R  # noqa: E402

URL = os.environ.get("VI_WHISPER_URL", "http://127.0.0.1:18090/transcribe")


def one(item, out):
    v = os.path.join(R.VIDEOS, item["video"])
    dst = os.path.join(out, os.path.basename(item["video"]) + ".tsv")
    if os.path.exists(dst):
        return item["item_id"], "cached"
    t = time.time()
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", v, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", "32k",
                        "-f", "ogg", "pipe:1"], capture_output=True)
    if not p.stdout:
        open(dst, "w").close()
        return item["item_id"], "no audio -> empty"
    for attempt in range(3):
        try:
            req = urllib.request.Request(URL, p.stdout, {"content-type": "application/octet-stream"})
            txt = urllib.request.urlopen(req, timeout=3600).read().decode()
            break
        except Exception as e:
            err = str(e)[:200]; time.sleep(10)
    else:
        return item["item_id"], "FAIL " + err
    open(dst + ".part", "w").write(txt); os.replace(dst + ".part", dst)
    return item["item_id"], f"{txt.count(chr(10))} segs, {len(p.stdout)/1e6:.1f} MB audio, {time.time()-t:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--set", default=""); ap.add_argument("-j", type=int, default=3)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(R.ITEMS)]
    if a.set == "hard60":
        hard = set(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hard60.json"))))
        items = [x for x in items if x["item_id"] in hard] + [x for x in items if x["item_id"] not in hard]
    with cf.ThreadPoolExecutor(a.j) as ex:
        for iid, msg in ex.map(lambda x: one(x, a.out), items):
            print(f"{iid:28s} {msg}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
