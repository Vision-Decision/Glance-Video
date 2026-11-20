#!/usr/bin/env python3
"""vdm server: Glance-4B as a batched per-frame visual detector over HTTP.

  python server.py [--port 8000] [--model Vision-Decision/Glance-4B-preview] [--batch 16] [--device cuda]

GET  /health -> {"ok": true, "model": ..., "device": ..., "frames_scored": n}
POST /score  -> per-image results. Two request forms:
  (a) contract form: {"query": str, "mode": "present"|"yesno"|"both", "images": [b64 jpeg], "want_box": bool}
      -> {"results": [{"p_present": float|None, "p_yes": float|None, "box": [x1,y1,x2,y2]|None}, ...]}
      "present" asks the box primitive with `query` as a referring expression (presence gate);
      "yesno" asks `query` as a yes/no question (if it does not end with "?", it becomes
      "Does this image show <query>?"); "both" asks both from ONE prefill.
  (b) generic form: {"questions": [{"type": "box", "text": ...} | {"type": "yesno", "question": ...} |
      {"type": "choice", "question": ..., "options": [...]}], "images": [...]}
      -> {"results": [[result per question] per image]}
  Optional: "max_vis_tokens" (default 640, as in natural-image training).
Every question about one image is answered from one shared prefill (Glance's block design); images are batched.
Results are cached by (image sha1, question, max_vis_tokens).
"""
import argparse, base64, hashlib, io, json, math, os, sys, threading, time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image

from glance import Glance, Box, YesNo, Choice
from glance.prompt import collate, make_item

M = None
ARGS = None
LOCK = threading.Lock()
CACHE = OrderedDict()
CACHE_MAX = 200_000
STATS = {"frames_scored": 0, "frames_cached": 0, "requests": 0, "gpu_sec": 0.0}


def yes_no_question(q):
    q = q.strip()
    return q if q.endswith("?") else f"Does this image show {q}?"


def build_questions(req):
    if req.get("questions"):
        qs = []
        for q in req["questions"]:
            t = q.get("type", "box")
            if t == "box":
                qs.append(Box(q["text"]))
            elif t == "yesno":
                qs.append(YesNo(q["question"]))
            elif t == "choice":
                qs.append(Choice(q["question"], list(q["options"])))
            else:
                raise ValueError(f"unsupported question type {t}")
        return qs, False
    mode = req.get("mode", "both")
    query = req["query"]
    qs = []
    if mode in ("present", "both"):
        qs.append(Box(query))
    if mode in ("yesno", "both"):
        qs.append(YesNo(yes_no_question(query)))
    if not qs:
        raise ValueError("mode must be present|yesno|both")
    return qs, True


def qkey(q):
    if isinstance(q, Box):
        return "box|" + q.text
    if isinstance(q, YesNo):
        return "yn|" + q.question
    return "ch|" + q.question + "|" + "\x1f".join(map(str, q.options))


@torch.no_grad()
def ask_batch(images, qs, max_vis_tokens):
    """Same as Glance.ask, for many images sharing one question list. -> list (per image) of list (per q) of dicts."""
    m = M
    blocks, labs = [], []
    for q in qs:
        if isinstance(q, Box):
            blocks.append(m.tok.box_block(q.text)); labs.append(None)
        elif isinstance(q, YesNo):
            ids, lab = m.tok.typed_block("yesno", q.question); blocks.append(ids); labs.append(lab)
        else:
            ids, lab = m.tok.typed_block("choice", q.question, [str(o) for o in q.options])
            blocks.append(ids); labs.append(lab)
    items = [make_item(m.tok, m.processor.image_processor, im, None, blocks, max_vis_tokens, m.cfg["max_upscale"])
             for im in images]
    out = m._forward(collate(items, m.tok))
    T = m.temperatures
    nq = len(qs)
    res = []
    for i, it in enumerate(items):
        W, H = it["orig_wh"]
        row = []
        for j, (q, lab) in enumerate(zip(qs, labs)):
            k = i * nq + j  # blocks are flattened item by item (checked against Glance.ask in selftest)
            if isinstance(q, Box):
                pl = float(out["presence_logit"][k]) / T.get("presence", 1.0)
                pp = 1 / (1 + math.exp(-pl))
                pi = float(torch.sigmoid(out["iou_logit"][k]))
                b = out["box"][k].clamp(0, 1) * torch.tensor([W, H, W, H], dtype=torch.float32)
                row.append({"type": "box", "p_present": round(pp, 4), "iou_prob": round(pi, 4),
                            "box": [round(float(v), 1) for v in b]})
            elif isinstance(q, YesNo):
                p = torch.softmax(out["label_logits"][k, lab] / T.get("noul", 1.0), -1)
                row.append({"type": "yesno", "p_yes": round(float(p[1]), 4)})
            else:
                p = torch.softmax(out["label_logits"][k, lab] / T.get("choice", 1.0), -1)
                row.append({"type": "choice", "probs": [round(float(v), 4) for v in p]})
        res.append(row)
    return res


def score(req):
    qs, contract = build_questions(req)
    mvt = int(req.get("max_vis_tokens", 640))
    imgs_b = [base64.b64decode(b) for b in req["images"]]
    shas = [hashlib.sha1(b).hexdigest() for b in imgs_b]
    keys = [qkey(q) for q in qs]
    out = [[None] * len(qs) for _ in imgs_b]
    todo = []
    with LOCK:
        for i, s in enumerate(shas):
            hit = [CACHE.get((s, k, mvt)) for k in keys]
            if all(h is not None for h in hit):
                out[i] = hit
                STATS["frames_cached"] += 1
            else:
                todo.append(i)
    t0 = time.time()
    for s in range(0, len(todo), ARGS.batch):
        idx = todo[s:s + ARGS.batch]
        ims = [Image.open(io.BytesIO(imgs_b[i])).convert("RGB") for i in idx]
        with LOCK:
            rows = ask_batch(ims, qs, mvt)
            for i, row in zip(idx, rows):
                out[i] = row
                for k, r in zip(keys, row):
                    CACHE[(shas[i], k, mvt)] = r
                while len(CACHE) > CACHE_MAX:
                    CACHE.popitem(last=False)
            STATS["frames_scored"] += len(idx)
    dt = time.time() - t0
    STATS["gpu_sec"] += dt
    STATS["requests"] += 1
    if contract:
        flat = []
        for row in out:
            d = {"p_present": None, "p_yes": None, "box": None}
            for r in row:
                if r["type"] == "box":
                    d["p_present"] = r["p_present"]
                    if req.get("want_box", True):
                        d["box"] = r["box"]
                        d["iou_prob"] = r["iou_prob"]
                else:
                    d["p_yes"] = r["p_yes"]
            flat.append(d)
        out = flat
    return {"results": out, "n": len(imgs_b), "scored": len(todo), "sec": round(dt, 3)}


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        p = self.path.rstrip("/")
        if p.endswith("/health"):
            return self._send(200, {"ok": True, "model": ARGS.model, "device": M.device, **STATS,
                                    "cache": len(CACHE)})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        p = self.path.rstrip("/")
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)))
            if p.endswith("/score"):
                return self._send(200, score(req))
            return self._send(404, {"error": "not found"})
        except Exception as e:
            return self._send(400, {"error": f"{type(e).__name__}: {e}"[:500]})


def selftest():
    """Batched path == Glance.ask on two synthetic images."""
    from PIL import ImageDraw
    a = Image.new("RGB", (448, 336), "white"); ImageDraw.Draw(a).ellipse((150, 100, 300, 250), fill="red")
    b = Image.new("RGB", (448, 336), "white")
    qs = [Box("a red circle"), YesNo("Is there a red circle?"), Choice("What color is the shape?", ["red", "blue", "green"])]
    batched = ask_batch([a, b], qs, 640)
    for im, row in zip([a, b], batched):
        ref = M.ask(im, qs)
        assert abs(ref[0].present_prob - row[0]["p_present"]) < 0.02, (ref[0], row[0])
        assert abs(ref[1].p_yes - row[1]["p_yes"]) < 0.02, (ref[1], row[1])
    print("selftest ok", json.dumps(batched), flush=True)


def main():
    global M, ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--model", default=os.environ.get("VDM_MODEL", "Vision-Decision/Glance-4B-preview"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--selftest", action="store_true")
    ARGS = ap.parse_args()
    t = time.time()
    M = Glance.from_pretrained(ARGS.model, device=ARGS.device)
    print(f"loaded {ARGS.model} on {M.device} in {time.time()-t:.0f}s", flush=True)
    if ARGS.selftest:
        selftest()
    ThreadingHTTPServer((ARGS.host, ARGS.port), H).serve_forever()


if __name__ == "__main__":
    main()
