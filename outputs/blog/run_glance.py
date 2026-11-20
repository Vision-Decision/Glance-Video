"""Run Glance on BBB frames sampled every 8 s over 0-472 s; write raw outputs as JSON."""
import json, subprocess, sys, time, os
from glance import Glance, Box, YesNo
video, outdir, out = sys.argv[1], sys.argv[2], sys.argv[3]
times = list(range(0, 480, 8))
for t in times:
    p = os.path.join(outdir, f"t{t:03d}.jpg")
    if not os.path.exists(p):
        subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", video, "-frames:v", "1", "-q:v", "2", p], check=True)
m = Glance.from_pretrained("Vision-Decision/Glance-4B-preview")
qs = [Box("the big white rabbit"), Box("a flying squirrel"), Box("a red car"), YesNo("Is there a butterfly in this frame?")]
rows = []
t0 = time.time()
for t in times:
    s = time.time()
    r = m.ask(os.path.join(outdir, f"t{t:03d}.jpg"), qs)
    rows.append({"t": t, "secs": round(time.time() - s, 2), "results": [dict(vars(x)) for x in r]})
    print(t, rows[-1]["secs"], [getattr(x, "present_prob", getattr(x, "p_yes", None)) for x in r], flush=True)
json.dump({"model": "Vision-Decision/Glance-4B-preview", "device": str(m.device), "video": "Big Buck Bunny (archive.org BigBuckBunny_124, big_buck_bunny_720p_surround.mp4, sha1 3bdcef855ca17af11e96b8f8dbaeb4d3e54b996a, 640x360)",
           "sample": "one frame every 8 s, 0-472 s", "questions": [str(q) for q in qs], "rows": rows, "total_secs": round(time.time() - t0, 1)},
          open(out, "w"), indent=1, default=list)
print("done", out)
