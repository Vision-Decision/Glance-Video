#!/usr/bin/env python3
"""vdm scan server: Glance(-Video) behind the vdm v1 HTTP API, backed by the scan engine (scan/engine.py).

What changes vs server_gv.py: clips are encoded ONCE (prefix state kept per clip in an ImageBank); every question over
those clips is one branch forward over all of them (no re-encoding, no decoding). Answers equal scoring each clip alone.

  python video/vdm/gv/scan_server.py --ckpt /content/ckpt/glance-video-v0/ckpt_last.pt --calib CALIB.json --port 8002
  GET  /health
  POST /score_clips  (unchanged vdm1 API) {"question"|"questions", "kind", "options", "dt", "clips": [[b64 jpeg]...]}
       -> {"results", "sec", "cached": bool}. The bank is keyed by a hash of the clips: the vdm1 client re-sends the
       same chunks for every question about the same video span, so follow-up questions skip the vision tower + prefill.
  POST /encode {"video": key, "clips": [[b64 jpeg]...], "times": [[t0, t1], ...], "dt": 0.5}  -> {"n", "sec"}
       or {"video": key, "frames": [b64 jpeg...], "K": 4, "stride": 2, "times": [...], "dt": 0.5}: frames sent once,
       clips = frames[i:i+K] for i = 0, stride, ... (half the upload of the clip form for K=4, stride 2)
  POST /reps   {"frames": [b64 jpeg...] at "fps" (8), "action": "..."} -> repetition count with the trained window head:
       non-overlapping 4-s windows (32 frames), "how many repetitions ... are completed in this clip" (0..12), summed
  POST /scan   {"video": key, "query": "Does ... in this clip?", "ops": ["curve","when","count","ever"],
                "alpha": 0.1, "q": 0.1}
       -> {"curve": [{t0,t1,p}], "ever": {...guarantee}, "when": {intervals}, "count": {...}, "sec", "n_clips"}
Calibration (--calib, from scan/pilot_charades.py + scan/video_ops.Calib.to_json) makes `ever` / `when` / `count`
come with their guarantees; without it, only `curve` is served and `ever` falls back to max p > 0.5 (no guarantee).
"""
import argparse, base64, hashlib, io, json, os, sys, threading, time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

ROOT = os.environ.get("DV_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
import torch  # noqa: E402
from scan import video_ops as VO  # noqa: E402

ENG, CAL = None, None
LOCK = threading.Lock()
BANKS = OrderedDict()            # key -> (bank, times)
OWNERS = {}                      # key -> owner tag (e.g. "<run>.<item>"); POST /release {"owner"} drops its banks
MAX_BANK_GB = float(os.environ.get("SCAN_MAX_BANK_GB", "150"))   # GPU + host RAM
STATS = {"clips_encoded": 0, "queries": 0, "gpu_sec": 0.0, "bank_hits": 0, "reps": 0}
REQLOG = os.environ.get("SCAN_REQLOG", "/content/scan_requests.jsonl")


def frames_to_clips(frames, K, stride):
    n = len(frames)
    return [frames[s:s + K] + [frames[-1]] * max(0, s + K - n) for s in range(0, max(1, n - K + 1), stride)]


def decode_clips(clips):
    out = []
    for c in clips:
        fr = [Image.open(io.BytesIO(base64.b64decode(f))).convert("RGB") for f in c]
        w = min(f.size[0] for f in fr); h = min(f.size[1] for f in fr)
        out.append(np.stack([np.asarray(f)[:h, :w] for f in fr]))
    return out


GPU_BANK_GB = float(os.environ.get("SCAN_GPU_BANK_GB", "50"))   # banks above this move to host RAM (LRU), not dropped
CLIP_BYTES = 56 * 2**20          # ~bank bytes per 4-frame clip (GDN states dominate); used to make room before encoding


def _move(bank, dev):
    """move every tensor of an ImageBank to dev (graphs are dropped: they belong to the old buffers)."""
    nb = lambda t: t.to(dev, non_blocking=False) if torch.is_tensor(t) else t
    bank.cache = [tuple(nb(x) for x in c) for c in bank.cache]
    for k in ("p_valid", "q_start", "vis", "vis_valid", "centers", "ck", "grid_hw"):
        setattr(bank, k, nb(getattr(bank, k)))
    bank._graphs = {}
    return bank


def _on_gpu(bank):
    return bank.p_valid.is_cuda


def _gpu_bytes(skip=None):
    return sum(b.nbytes() for k, (b, _) in BANKS.items() if k != skip and _on_gpu(b))


def _make_room(need, keep=None):
    """demote least recently used GPU banks to host RAM until `need` bytes fit in the GPU budget; drop banks beyond
    the total budget (MAX_BANK_GB, GPU + host)."""
    for k in list(BANKS):
        if _gpu_bytes(skip=keep) + need <= GPU_BANK_GB * 2**30:
            break
        if k != keep and _on_gpu(BANKS[k][0]):
            _move(BANKS[k][0], "cpu"); STATS["demoted"] = STATS.get("demoted", 0) + 1
    torch.cuda.empty_cache()
    while sum(b.nbytes() for b, _ in BANKS.values()) + need > MAX_BANK_GB * 2**30 and len(BANKS) > (1 if keep else 0):
        k = next(k for k in BANKS if k != keep)
        BANKS.pop(k); OWNERS.pop(k, None); STATS["dropped"] = STATS.get("dropped", 0) + 1


def bank_for(key, clips_b64=None, dt=0.5, times=None):
    """cached bank for key (promoted to the GPU if it was demoted), else encode clips_b64. Two-tier LRU."""
    if key in BANKS:
        BANKS.move_to_end(key)
        STATS["bank_hits"] += 1
        bank = BANKS[key][0]
        if torch.cuda.is_available() and not _on_gpu(bank):   # CPU-only host (tests): banks live on the CPU
            _make_room(bank.nbytes(), keep=key)
            _move(bank, "cuda"); STATS["promoted"] = STATS.get("promoted", 0) + 1
        return bank, BANKS[key][1], False
    if clips_b64 is None:
        raise KeyError(f"unknown video '{key}': POST /encode first")
    _make_room(len(clips_b64) * CLIP_BYTES)
    ENG.dt = dt
    clips = decode_clips(clips_b64)
    bank = ENG.encode(clips, chunk=64)
    STATS["clips_encoded"] += len(clips)
    times = times or [[i * 1.0, i * 1.0 + dt * (len(c) - 1)] for i, c in enumerate(clips)]
    BANKS[key] = (bank, times)
    return bank, times, True


READOUT = "yes"      # --readout short: the ~10-token scan block (model trained for it, scan/distill_*)


def yes_logits(bank, instr):
    q = dict(type="noul", instr=instr, block="short") if READOUT == "short" else dict(type="noul", instr=instr)
    tl = ENG.query(bank, q)["typed_logits"]
    return (tl[:, 1] - tl[:, 0]).numpy()


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
            return self._reply(200, {"ok": True, "version": "scan-v1", "calibrated": CAL is not None,
                                     "banks": len(BANKS), "bank_gb": round(sum(b.nbytes() for b, _ in BANKS.values()) / 2**30, 2),
                                     "gpu_bank_gb": round(_gpu_bytes() / 2**30, 2),
                                     "readout": READOUT, **STATS})
        self._reply(404, {"error": "not found"})

    def do_POST(self):
        p = self.path.rstrip("/")
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)))
            t = time.time()
            with LOCK, torch.no_grad():
                if p == "/score_clips":
                    key = hashlib.sha1("".join(f for c in req["clips"] for f in c).encode()).hexdigest()
                    bank, _, new = bank_for(key, req["clips"], float(req.get("dt", 0.5)))
                    qs = req.get("questions") or [dict(type="noul", instr=req["question"]) if req.get("kind", "yesno") == "yesno"
                                                  else dict(type="choice", instr=req["question"], options=req["options"])]
                    rows = [[] for _ in range(bank.N)]
                    for q in qs:
                        tl = ENG.query(bank, q)["typed_logits"]
                        for i in range(bank.N):
                            lg = tl[i][torch.isfinite(tl[i])]
                            pr = torch.softmax(lg, -1)
                            rows[i].append(float(pr[1]) if q["type"] == "noul" else [float(v) for v in pr])
                    STATS["queries"] += len(qs)
                    res = [r[0] for r in rows] if not req.get("questions") else rows
                    out = {"results": res, "cached": not new}
                elif p == "/encode":
                    clips = req.get("clips") or frames_to_clips(req["frames"], int(req.get("K", 4)), int(req.get("stride", 2)))
                    bank, times, new = bank_for(req["video"], clips, float(req.get("dt", 0.5)), req.get("times"))
                    if req.get("owner"):
                        OWNERS[req["video"]] = req["owner"]
                    out = {"n": bank.N, "cached": not new}
                elif p == "/release":
                    # drop the banks of a finished agent run so they never evict banks of runs still in flight
                    own = req.get("owner") or ""
                    gone = [k for k, o in OWNERS.items() if own and (o == own or o.startswith(own + "."))]   # "<run>" releases all its items
                    for k in gone:
                        BANKS.pop(k, None); OWNERS.pop(k, None)
                    torch.cuda.empty_cache()
                    out = {"released": len(gone)}
                elif p == "/reps":
                    fps = float(req.get("fps", 8.0))
                    fr = decode_clips([req["frames"]])[0]
                    W = int(round(4 * fps))
                    wins = [fr[s:s + W] for s in range(0, len(fr), W) if len(fr[s:s + W]) >= max(8, W // 4)]
                    if not wins:
                        raise ValueError("too few frames for one 4-s window")
                    dt0, mv0 = ENG.dt, ENG.clip_max_vis
                    ENG.dt, ENG.clip_max_vis = 1.0 / fps, 48
                    try:
                        _make_room(len(wins) * CLIP_BYTES * 2)
                        bank = ENG.encode(wins, chunk=16)
                        act = req["action"].strip().rstrip("?.")
                        tl = ENG.query(bank, dict(type="choice", options=[str(i) for i in range(13)],
                                                  instr=f"How many repetitions of this repeated action ({act}) are completed in this clip?"))["typed_logits"]
                    finally:
                        ENG.dt, ENG.clip_max_vis = dt0, mv0
                    pr = torch.softmax(tl[:, :13].float(), -1).numpy()
                    ev = pr @ np.arange(13)
                    t0 = float(req.get("t0", 0.0))
                    out = {"count": round(float(ev.sum()), 2), "count_argmax": int(pr.argmax(1).sum()),
                           "windows": [dict(t0=round(t0 + i * 4.0, 1), t1=round(t0 + i * 4.0 + len(w) / fps, 1), expected=round(float(e), 2))
                                       for i, (w, e) in enumerate(zip(wins, ev))]}
                    STATS["reps"] += 1
                    del bank
                elif p == "/scan":
                    bank, times, _ = bank_for(req["video"])
                    s = yes_logits(bank, req["query"])
                    STATS["queries"] += 1
                    ops = req.get("ops", ["curve", "ever", "when"])
                    a, q = float(req.get("alpha", 0.1)), float(req.get("q", 0.1))
                    out = {"n_clips": bank.N}
                    if "curve" in ops:
                        out["curve"] = VO.curve(s, times, CAL)
                    if CAL is None:
                        out["ever"] = {"ever": bool((s > 0).any()), "method": "naive p>0.5", "guarantee": None}
                    else:
                        if "ever" in ops:
                            out["ever"] = VO.ever(s, CAL, a)
                        if "when" in ops:
                            out["when"] = VO.when(s, times, CAL, q, a)
                        if "count" in ops:
                            out["count"] = VO.count(s, times, CAL, q, a, method=req.get("count_method", "fwer"))
                        if "scores" in ops:   # raw per-clip evidence for client-side top-k: calibrated p + conformal p-value
                            from scan import setlevel as SL
                            out["scores"] = dict(times=times, p=np.round(1 / (1 + np.exp(-s / CAL.T)), 4).tolist(),
                                                 pval=np.round(SL.conformal_pvalues(s, CAL.pool, presorted=True), 5).tolist())
                else:
                    return self._reply(404, {"error": "not found"})
            sec = time.time() - t
            STATS["gpu_sec"] += sec
            out["sec"] = round(sec, 4)
            try:
                with open(REQLOG, "a") as f:
                    f.write(json.dumps({"ts": t, "path": p, "sec": round(sec, 4), "n": out.get("n_clips", out.get("n")),
                                        "cached": out.get("cached"), "video": str(req.get("video", ""))[:12]}) + "\n")
            except Exception:
                pass
            self._reply(200, out)
        except Exception as e:
            self._reply(400, {"error": f"{type(e).__name__}: {e}"[:300]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--calib", default="")
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--max-vis", type=int, default=96)
    ap.add_argument("--graphs", action="store_true")
    ap.add_argument("--readout", default="yes", choices=["yes", "short"])
    a = ap.parse_args()
    from scan.pilot_charades import load_glance
    ENG, _ = load_glance(a.ckpt)
    ENG.clip_max_vis = a.max_vis
    READOUT = a.readout
    if a.graphs:
        ENG.graphs = True; ENG.fv.graphs = True
    if a.calib:
        CAL = VO.Calib.from_json(json.load(open(a.calib)))
    print("scan server ready", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
