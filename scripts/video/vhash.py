"""Perceptual frame hashes for video decontamination (Glance-Video).

frame_hashes(path, fps): ffmpeg-decoded grayscale frames -> 64-bit DCT pHash per frame (uint64), skipping near-uniform
frames (black/white/fades) that would match anything. HashIndex: multi-index (7 chunks; Hamming <= 6 => at least one
chunk equal) over the Video-Index frame hashes; `match(hashes)` returns {vi_video_id: n_frames_within_threshold}.
"""
import subprocess, numpy as np, cv2

SIDE = 32


def _frames(path, fps, max_frames=None, start=None, dur=None):
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "2"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", path]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-an", "-vf", f"fps={fps},scale={SIDE}:{SIDE}:flags=area,format=gray", "-f", "rawvideo", "-"]
    if max_frames:
        cmd[cmd.index("-f"):cmd.index("-f")] = ["-frames:v", str(max_frames)]
    out = subprocess.run(cmd, capture_output=True, timeout=3600).stdout
    n = len(out) // (SIDE * SIDE)
    return np.frombuffer(out[: n * SIDE * SIDE], np.uint8).reshape(n, SIDE, SIDE)


def phash(gray32):
    """gray32: [N,32,32] uint8 -> (hashes uint64 [N], informative bool [N])"""
    if len(gray32) == 0:
        return np.zeros(0, np.uint64), np.zeros(0, bool)
    hs, ok = [], []
    for g in gray32:
        f = g.astype(np.float32)
        d = cv2.dct(f)[:8, :8].flatten()
        med = np.median(d[1:])
        bits = (d > med).astype(np.uint64)
        bits[0] = 0
        hs.append(int((bits << np.arange(64, dtype=np.uint64)).sum()))
        ok.append(f.std() > 8.0)
    return np.array(hs, dtype=np.uint64), np.array(ok)


def frame_hashes(path, fps=1.0, **kw):
    h, ok = phash(_frames(path, fps, **kw))
    return h[ok]


_SHIFTS = [0, 9, 18, 27, 36, 45, 54]
_WIDTH = [9, 9, 9, 9, 9, 9, 10]


def _chunks(h):
    h = np.asarray(h, dtype=np.uint64)
    return [((h >> np.uint64(s)) & np.uint64((1 << w) - 1)).astype(np.int64) for s, w in zip(_SHIFTS, _WIDTH)]


def popcount64(x):
    x = np.asarray(x, dtype=np.uint64)
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)


class HashIndex:
    def __init__(self, hashes, owners):
        self.h = np.asarray(hashes, dtype=np.uint64)
        self.owner = np.asarray(owners)
        self.tables = []
        for c in _chunks(self.h):
            order = np.argsort(c, kind="stable")
            self.tables.append((c[order], order))

    def match(self, q, thr=6):
        """q: uint64 hashes of one candidate video -> dict owner -> number of query frames matching that owner"""
        res = {}
        if len(q) == 0 or len(self.h) == 0:
            return res
        qc = _chunks(q)
        for i in range(len(q)):
            cand = []
            for (keys, order), c in zip(self.tables, qc):
                lo, hi = np.searchsorted(keys, c[i], "left"), np.searchsorted(keys, c[i], "right")
                if hi > lo:
                    cand.append(order[lo:hi])
            if not cand:
                continue
            cand = np.unique(np.concatenate(cand))
            d = popcount64(self.h[cand] ^ np.uint64(q[i]))
            for o in set(self.owner[cand[d <= thr]].tolist()):
                res[o] = res.get(o, 0) + 1
        return res
