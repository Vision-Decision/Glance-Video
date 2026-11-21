"""Feed decoded video into a Stream: ViT in batches, `chunk` groups per LM forward (chunk=1 is the strict per-frame mode),
optional callback after every write (readouts)."""
import time
import numpy as np
import torch
from . import vio


def feed(st, frames, times, chunk=1, t_offset=0.0, prev=None, cb=None, every=None, vit_batch=256, feats=False):
    """frames uint8 (T,3,S,S), times (T,) local seconds; stream time = t_offset + times. cb(stream_time) is called after the
    write that crosses each multiple of `every` seconds (or after every write if every is None).
    Returns (last frame, timing dict, feats list)."""
    tm = dict(vit=0.0, lm=0.0, n=0, step_ms=[])
    out_feats = []
    T = len(times)
    nxt = None if every is None else (np.floor((t_offset + times[0]) / every) + 1) * every if T else None
    for a in range(0, T, vit_batch):
        fr = frames[a:a + vit_batch]
        pr = vio.pairs(fr, prev if a == 0 else frames[a - 1])
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        vis, grid = st.encode(pr)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        tm["vit"] += time.perf_counter() - t0
        for b in range(0, len(fr), chunk):
            ts = [float(t_offset + x) for x in times[a + b:a + min(b + chunk, len(fr))]]
            t1 = time.perf_counter()
            f = st.write(vis[b:b + chunk], ts, grid, return_feats=feats)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            dt = time.perf_counter() - t1
            tm["lm"] += dt; tm["n"] += len(ts); tm["step_ms"].append(dt * 1e3)
            if feats:
                out_feats.append((ts, f))
            if cb is not None:
                if every is None:
                    cb(ts[-1])
                else:
                    if ts[-1] + 1e-6 >= nxt:
                        cb(ts[-1]); nxt = (np.floor(ts[-1] / every + 1e-6) + 1) * every
    return (frames[-1] if T else prev), tm, out_feats
