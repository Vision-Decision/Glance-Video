#!/usr/bin/env python3
"""vdm v1 server: clip-level yes/no and multiple-choice scoring (see core.py).

  python server_v1.py --port 8001
  GET  /health
  POST /score_clips {"question": str, "kind": "yesno"|"choice", "options": [...], "backend": "glance"|"qwen",
                     "clips": [[b64 jpeg frame, ...], ...]}       (frames of one clip in time order, same size)
       -> {"results": [p_yes | [option probs]], "sec": float}
"""
import argparse, base64, io, json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

from core import ClipScorer

S = None
LOCK = threading.Lock()
STATS = {"clips": 0, "requests": 0, "gpu_sec": 0.0}


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _reply(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            return self._reply(200, {"ok": True, "version": "v1-clip", **STATS})
        self._reply(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/score_clips":
            return self._reply(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)))
            clips = [[Image.open(io.BytesIO(base64.b64decode(f))).convert("RGB") for f in c] for c in req["clips"]]
            t = time.time()
            with LOCK:
                res = S.score(clips, req["question"], req.get("backend", "glance"), req.get("kind", "yesno"),
                              req.get("options"))
            dt = time.time() - t
            STATS["clips"] += len(clips); STATS["requests"] += 1; STATS["gpu_sec"] += dt
            self._reply(200, {"results": res, "sec": round(dt, 3)})
        except Exception as e:
            self._reply(400, {"error": f"{type(e).__name__}: {e}"[:300]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--max-vis-tokens", type=int, default=256)
    a = ap.parse_args()
    S = ClipScorer(max_vis_tokens=a.max_vis_tokens)
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
