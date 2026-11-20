#!/usr/bin/env python3
"""vdm v1 (Glance-Video) server. Same API as video/vdm/v1/server_v1.py, so the existing `vdm1` client works unchanged
(point VDM1_URL at this server); the `backend` field is ignored (always the Glance-Video checkpoint, or base Qwen3.5-4B
with --base). Extra: `questions` (several questions about the same clips share one prefix) and `dt`.

  python server_gv.py --ckpt /content/out/gv0/ckpt_final.pt --port 8002 [--base]
  GET  /health
  POST /score_clips {"question": str, "kind": "yesno"|"choice", "options": [...], "dt": 0.5,
                     "clips": [[b64 jpeg frame, ...], ...]}             -> {"results": [p_yes | [option probs]], "sec"}
  POST /score_clips {"questions": [{"type": "noul"|"choice", "instr": str, "options": [...]}, ...], "clips": [...]}
                                                                          -> {"results": [[per question] per clip], "sec"}
Scan interface (encode once, query many times; see core_gv.ClipBackend):
  POST /prepare {"video_id": str, "clips": [[b64 jpeg, ...], ...], "t0": [clip start s, ...], "dt": 0.5}
  POST /scan    {"video_id": str, "questions": [...]}       -> {"t0": [...], "results": [[per question] per clip], "sec"}
  POST /drop    {"video_id": str}
--backend module:Class swaps in another ClipBackend (e.g. a cached-prefix scan engine) with the same constructor args.
"""
import argparse, base64, io, json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image


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
            return self._reply(200, {"ok": True, "version": "v1-glance-video", **STATS})
        self._reply(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.rstrip("/")
        if path not in ("/score_clips", "/prepare", "/scan", "/drop"):
            return self._reply(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)))
            if path in ("/scan", "/drop"):
                t = time.time()
                with LOCK:
                    out = S.scan(req["video_id"], req["questions"]) if path == "/scan" else (S.drop(req["video_id"]) or {})
                sec = time.time() - t
                STATS["requests"] += 1; STATS["gpu_sec"] += sec
                return self._reply(200, dict(out, sec=round(sec, 3)))
            clips = []
            for c in req["clips"]:
                fr = [np.asarray(Image.open(io.BytesIO(base64.b64decode(f))).convert("RGB")) for f in c]
                h = min(x.shape[0] for x in fr); w = min(x.shape[1] for x in fr)
                clips.append(np.stack([x[:h, :w] for x in fr]))
            dt = float(req.get("dt", 0.5))
            t = time.time()
            with LOCK:
                if path == "/prepare":
                    res = S.prepare(req["video_id"], clips, t0s=req.get("t0"), dt=dt)
                elif req.get("questions"):
                    res = S.score_multi(clips, req["questions"], dt=dt)
                else:
                    res = S.score(clips, req["question"], req.get("kind", "yesno"), req.get("options"), dt=dt)
            sec = time.time() - t
            STATS["clips"] += len(clips); STATS["requests"] += 1; STATS["gpu_sec"] += sec
            self._reply(200, {"results": res, "sec": round(sec, 3)})
        except Exception as e:
            self._reply(400, {"error": f"{type(e).__name__}: {e}"[:300]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--max-vis", type=int, default=96)
    ap.add_argument("--backend", default="core_gv:GVScorer", help="module:Class implementing core_gv.ClipBackend")
    a = ap.parse_args()
    import importlib
    mod, cls = a.backend.split(":")
    S = getattr(importlib.import_module(mod), cls)(a.ckpt, base=a.base, max_vis=a.max_vis)
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
