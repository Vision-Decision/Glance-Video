#!/usr/bin/env python3
"""Per-video preprocessing for the paired Video-Index runs (run once per video, shared by every run and arm).

  python3 prep_vi.py --transcripts DIR [--set hard60 | --ids a,b] -j 6     # faster-whisper small -> DIR/<video>.tsv
  python3 prep_vi.py --frames [--set hard60] -j 6                           # vdmscan whole-video frames -> scan_cache/<vid>/
Both run inside the agent image (same ffmpeg / faster-whisper / client as the agents). Videos never leave this host.
"""
import argparse, concurrent.futures as cf, json, os, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as R  # noqa: E402

WHISPER = r'''
import sys, time
from faster_whisper import WhisperModel
t = time.time()
m = WhisperModel("/opt/models/faster-whisper/small", device="cpu", compute_type="int8", cpu_threads=int(sys.argv[2]))
segs, info = m.transcribe(sys.argv[1], beam_size=1, vad_filter=True, condition_on_previous_text=False)
with open("/out/t.tsv.part", "w") as f:
    for s in segs:
        f.write(f"{s.start:.1f}\t{s.end:.1f}\t{s.text.strip()}\n")
import os; os.replace("/out/t.tsv.part", "/out/t.tsv")
print(f"{info.language} {info.duration:.0f}s in {time.time()-t:.0f}s")
'''


def transcribe(item, out, threads):
    v = os.path.join(R.VIDEOS, item["video"])
    dst = os.path.join(out, os.path.basename(item["video"]) + ".tsv")
    if os.path.exists(dst):
        return item["item_id"], "cached"
    tmp = dst + ".d"
    os.makedirs(tmp, exist_ok=True); os.chmod(tmp, 0o777)
    p = R.sh(["docker", "run", "--rm", "--network", "none", f"--cpus={threads}", "--memory=16g", "-v", f"{v}:/v.mp4:ro",
              "-v", f"{R.MODELS}:{R.WHISPER_IN}:ro", "-v", f"{tmp}:/out", "--entrypoint", "python3", R.IMAGE, "-c", WHISPER,
              "/v.mp4", str(threads)])
    if p.returncode == 0 and os.path.exists(os.path.join(tmp, "t.tsv")):
        os.replace(os.path.join(tmp, "t.tsv"), dst)
    elif "does not contain any stream" in p.stderr or "audio" in p.stderr.lower() and "stream" in p.stderr.lower():
        open(dst, "w").close()               # no audio track: empty transcript
    else:
        return item["item_id"], "FAIL " + p.stderr[-300:]
    R.sh(["rm", "-rf", tmp])
    return item["item_id"], p.stdout.strip()


def frames(item, cpus):
    v = os.path.join(R.VIDEOS, item["video"])
    vid = R.video_id(v)
    vc = os.path.join(R.CACHE, vid)
    os.makedirs(vc, exist_ok=True); os.chmod(vc, 0o777)
    p = R.sh(["docker", "run", "--rm", "--network", "none", f"--cpus={cpus}", "--memory=24g", "-v", f"{v}:/work/video.mp4:ro",
              "-v", f"{vc}:/cache/vdmscan/{vid}", "-v", f"{R.SCAN_CLIENT}:/usr/local/bin/vdmscan:ro",
              "-e", f"VDMSCAN_CACHE=/cache/vdmscan/{vid}", "-e", "VDMSCAN_LOG=/dev/null", "--entrypoint", "vdmscan", R.IMAGE,
              "warm", "/work/video.mp4", "--extract-only"])
    return item["item_id"], (p.stdout.strip() or p.stderr[-300:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", default="")
    ap.add_argument("--frames", action="store_true")
    ap.add_argument("--set", default="")
    ap.add_argument("--ids", default="")
    ap.add_argument("-j", type=int, default=6)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(R.ITEMS)]
    if a.set == "hard60":
        hard = set(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hard60.json"))))
        items = [x for x in items if x["item_id"] in hard]
    if a.ids:
        want = set(a.ids.split(","))
        items = [x for x in items if x["item_id"] in want]
    t = time.time()
    with cf.ThreadPoolExecutor(a.j) as ex:
        if a.transcripts:
            os.makedirs(a.transcripts, exist_ok=True)
            for iid, msg in ex.map(lambda x: transcribe(x, a.transcripts, a.threads), items):
                print(f"{iid:28s} {msg}", flush=True)
        if a.frames:
            for iid, msg in ex.map(lambda x: frames(x, a.threads), items):
                print(f"{iid:28s} {msg[-160:]}", flush=True)
    print(f"done {len(items)} in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
