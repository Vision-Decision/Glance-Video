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


def feed_batch(st, vids, chunk, on_end, on_step=None):
    """B videos as one lockstep batched stream from t=0 (all on the same k/fps grid). vids: list of (frames, times).
    on_end(i, row) is called right after video i's last group is written (row = its current batch row); the row is
    then dropped. on_step(times_written, active_ids) after every write. Returns timing."""
    st.reset(batch=len(vids))
    active = list(range(len(vids))); pos = 0; tm = dict(vit=0.0, lm=0.0)
    lens = [len(t) for _, t in vids]
    while active:
        c = min(chunk, min(lens[i] - pos for i in active))
        prs = torch.stack([vio.pairs(vids[i][0][pos:pos + c], vids[i][0][pos - 1] if pos else None) for i in active])
        t0 = time.perf_counter()
        vis, grid = st.encode(prs.flatten(0, 1))
        vis = vis.view(len(active), c, *vis.shape[1:])
        t1 = time.perf_counter()
        ts = [float(x) for x in vids[active[0]][1][pos:pos + c]]
        st.write(vis, ts, grid)
        tm["vit"] += t1 - t0; tm["lm"] += time.perf_counter() - t1
        pos += c
        if on_step is not None:
            on_step(ts, active)
        ended = [r for r, i in enumerate(active) if lens[i] == pos]
        for r in ended:
            on_end(active[r], r)
        if ended:
            keep = [r for r in range(len(active)) if r not in ended]
            active = [active[r] for r in keep]
            if active:
                st.drop(keep)
    return tm
