import json, sys, time, os, glob
from glance import Glance, Box, YesNo
d = sys.argv[1]; out = sys.argv[2]
m = Glance.from_pretrained("Vision-Decision/Glance-4B-preview")
qs = [Box("the big white rabbit"), Box("a flying squirrel"), Box("a butterfly"), Box("a red car"), YesNo("Is there a butterfly in this frame?")]
rows = []
for f in sorted(glob.glob(os.path.join(d, "t*.jpg"))):
    t = int(os.path.basename(f)[1:4]); s = time.time()
    r = m.ask(f, qs)
    rows.append({"t": t, "secs": round(time.time() - s, 2), "results": [dict(vars(x)) for x in r]})
    print(t, rows[-1]["secs"], flush=True)
json.dump({"model": "Vision-Decision/Glance-4B-preview", "device": str(m.device), "video": "Big Buck Bunny, archive.org BigBuckBunny_124 big_buck_bunny_720p_surround.mp4 (sha1 3bdcef855ca17af11e96b8f8dbaeb4d3e54b996a, 640x360)", "sample": "one frame every 4 s, 0-476 s", "questions": [str(q) for q in qs], "rows": rows}, open(out, "w"), indent=1, default=list)
