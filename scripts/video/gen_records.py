"""Glance-Video training / eval records (CPU, gen-host).  Record format: dv/data/video.py.

Sources (after decontamination, decon/exclusions.json):
  k700val   Kinetics-700-2020 val clips (10 s), labels + motion stats (clip_stats.py)
  countix   Countix train/val repetition counts on Kinetics-700 clips
  charades  Charades train split (actions with intervals) + Charades-STA train sentences

Generators (every record = one clip presentation + several typed questions; targets are exact by construction
or from annotations):
  K700   aot      forward / reversed presentation -> "playing forward?" (high object-motion windows only)
         camera   camera-motion direction from the similarity fit of the optical flow (flipped when reversed)
         action   "Is someone <label> in this clip?" (+ a negative label), forward presentations only
         speed    0.5x / 1x / 2x / 4x playback from the 8 fps clip (frame stride vs the timestamps)
         freeze   one frame repeated -> "does anything move?" = no; moving clip -> yes
         order    two segments with a gap, shown in order or swapped -> "is the later part shown after the cut?"
         synrep   a short segment repeated k times (loop or forward-backward) -> "how many times ...?"
  Countix count   real repetition counts (choice over 5 numbers) + "exactly k times?" yes/no
  Charades event  window yes/no: positives (>= 60% covered), hard negatives = actions of the SAME video in other
                  windows, reversal antonyms, other actions with the same object; random others
           rev    reversed presentation of an action with a reversal antonym (open<->close, put<->take, sit<->stand,
                  dress<->undress, light on<->off) -> the label flips
           order  two actions inside the window -> which happens first (flips when reversed)
           sta    Charades-STA sentences as yes/no (same coverage rule)

  python gen_records.py --root ~/gv --out ~/gv/records [--scale 1.0]
Eval files (held-out K700 ytids with hash % 20 == 0; Charades test split): eval_aot, eval_camera, eval_speed,
eval_charades_event, eval_countix (Countix val).
"""
import argparse, csv, glob, hashlib, json, math, os, random
from collections import defaultdict
import numpy as np

SFPS = 8  # stored clip fps
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

CAM_OPTS = {"pan_left": "The camera pans to the left.", "pan_right": "The camera pans to the right.",
            "tilt_up": "The camera tilts up.", "tilt_down": "The camera tilts down.",
            "fwd": "The camera moves forward or zooms in.", "back": "The camera moves backward or zooms out.",
            "static": "The camera stays still."}
CAM_FLIP = {"pan_left": "pan_right", "pan_right": "pan_left", "tilt_up": "tilt_down", "tilt_down": "tilt_up",
            "fwd": "back", "back": "fwd", "static": "static"}
SPEED_LEVELS = ["slow motion (about half speed)", "normal speed", "about twice as fast as normal",
                "about four times as fast as normal"]
REV_PAIRS = [("c006", "c008"), ("c025", "c027"), ("c039", "c041"), ("c046", "c048"), ("c089", "c090"),
             ("c112", "c113"), ("c142", "c143"), ("c001", "c002"), ("c017", "c018"), ("c022", "c023"),
             ("c028", "c030"), ("c034", "c035"), ("c042", "c043"), ("c049", "c050"), ("c054", "c056"),
             ("c055", "c057"), ("c062", "c063"), ("c068", "c069"), ("c071", "c073"), ("c077", "c079"),
             ("c099", "c100"), ("c109", "c110"), ("c116", "c117"), ("c119", "c120"), ("c148", "c155"),
             ("c151", "c154"), ("c104", "c105")]
REV = {a: b for x, y in REV_PAIRS for a, b in ((x, y), (y, x))}

YN_T = ["Is someone {a} in this clip?", "Does this clip show someone {a}?", "Is a person {a} at some point in this clip?"]
STA_T = ["Does this happen in this clip: {s}?", "Is this shown in the clip: {s}?"]


def h20(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % 20


def rel(p, root):
    """-> "gv/clips/..." (relative to data_root); index files from another host keep their absolute paths"""
    i = p.find("/gv/clips/")
    return p[i + 1:] if i >= 0 else os.path.relpath(p, os.path.dirname(root.rstrip("/")))


def plen_est(n_frames, w, h, max_vis):
    from_w, from_h = w, h
    s = min(1.0, math.sqrt(max_vis * 1024 / (from_w * from_h)))
    gw, gh = max(1, int(from_w * s) // 32), max(1, int(from_h * s) // 32)
    return 4 + math.ceil(n_frames / 2) * (gw * gh + 9)


def shuffle_choice(rng, opts, correct):
    """opts list of str, correct index -> (options, one-hot target) in random order"""
    order = list(range(len(opts)))
    rng.shuffle(order)
    return [opts[i] for i in order], [1.0 if i == correct else 0.0 for i in order]


def phrase(cls_name):
    s = cls_name.strip()
    for pre in ("Someone is ", "Someone "):
        if s.startswith(pre):
            s = s[len(pre):]
    s = s.replace("Wash a dish/dishes", "washing a dish").replace("/", " or ")
    return s[0].lower() + s[1:]


# ------------------------------------------------------------------------------------------------ K700 helpers
def cam_label(st, a, b):
    """stats steps [a, b) (0.25 s each) -> camera label or None (ambiguous)"""
    tx, ty, s = (np.array(st[k][a:b]) for k in ("tx", "ty", "s"))
    if len(tx) < 4:
        return None
    if (np.abs(tx) < 0.002).mean() > 0.9 and (np.abs(ty) < 0.002).mean() > 0.9 and (np.abs(s) < 0.002).mean() > 0.9:
        return "static"
    comp = {"x": (tx, 0.008), "y": (ty, 0.008), "s": (s, 0.006)}
    score = {k: abs(v.mean()) / t for k, (v, t) in comp.items()}
    k = max(score, key=score.get)
    others = [score[o] for o in score if o != k]
    v = comp[k][0]
    cons = (np.sign(v) == np.sign(v.mean())).mean()
    if score[k] < 1.5 or max(others) > 0.5 * score[k] or cons < 0.8:
        return None
    m = v.mean()
    if k == "x":
        return "pan_right" if m < 0 else "pan_left"       # content moves left => camera pans right
    if k == "y":
        return "tilt_up" if m > 0 else "tilt_down"        # content moves down => camera tilts up
    return "fwd" if m > 0 else "back"


def motion(st, a, b):
    r = np.array(st["res"][a:b]); d = np.array(st["diff"][a:b])
    return float(r.mean()) if len(r) else 0.0, float(d.mean()) if len(d) else 0.0


class Gen:
    def __init__(self, a):
        self.a, self.root = a, a.root
        self.rng = random.Random(a.seed)
        ex = json.load(open(f"{a.root}/decon/exclusions.json"))
        self.ex_ytid = set(ex["ids"].get("kinetics_ytid", [])) | set(ex["ids"].get("youtube", []))
        self.ex_charades = set(ex["ids"].get("charades", []))
        self.ex_paths = {c["rel"] for c in ex["phash"]["contaminated"]}       # "gv/clips/..." (host independent)
        for f in glob.glob(f"{a.root}/decon/clip_exclusions_*.json"):        # also any newer local pHash results
            self.ex_paths |= {c.get("rel") or rel(c["path"], a.root) for c in json.load(open(f))["contaminated"]}
        self.ex_sha_raw = set(ex.get("sha256_matches", []))                  # raw source files (none so far)
        self.n_ex = defaultdict(int)
        self.out = defaultdict(list)

    def emit(self, split, rec):
        rec["questions"] = [q for q in rec["questions"] if q]
        if rec["questions"]:
            self.out[split].append(rec)

    def rec(self, src, path, frames, dt, w, h, max_vis, qs, **kw):
        return dict(src=src, video=rel(path, self.root), frames=frames, dt=dt, max_vis=max_vis,
                    plen=plen_est(len(frames), w, h, max_vis), questions=qs, **kw)

    # -------------------------------------------------------------------------------------------- K700
    def k700(self):
        rng, a = self.rng, self.a
        idx = {json.loads(l)["path"]: json.loads(l) for l in open(f"{a.root}/clips/k700val/index.jsonl")}
        stats = {}
        for l in open(f"{a.root}/stats/k700val.jsonl"):
            d = json.loads(l); stats[d["path"]] = d
        labels = sorted({c["label"] for c in idx.values()})
        res_all = np.array([np.mean(s["res"]) for s in stats.values() if s["res"]])
        hi_res = float(np.quantile(res_all, 0.5))
        print("k700 clips", len(idx), "with stats", len(stats), "res median", hi_res, flush=True)
        for p, c in idx.items():
            st = stats.get(p)
            if st is None or len(st["res"]) < 8:
                continue
            if c["ytid"] in self.ex_ytid or rel(p, self.root) in self.ex_paths:
                self.n_ex["k700"] += 1
                continue
            hh = h20(c["ytid"])
            split = "eval" if hh == 0 else "train"
            calib = hh == 1
            N = c["n"]; W, H = c["w"], c["h"]
            mv = 96 if rng.random() < 0.5 else 64
            # ---- aot + camera + action (one or two presentations)
            for _ in range(1 if split == "eval" else a.k700_pres):
                fps = rng.choice([2, 4])
                stride = SFPS // fps
                L = rng.choice([2, 3, 4, 6, 8]) if fps == 2 else rng.choice([2, 3, 4])
                n = min(16, int(L * fps))
                span = (n - 1) * stride + 1
                if span > N:
                    continue
                i0 = rng.randrange(0, N - span + 1)
                fr = list(range(i0, i0 + span, stride))
                sa, sb = i0 // 2, (i0 + span - 1) // 2
                res, diff = motion(st, sa, sb)
                rev = res >= hi_res and rng.random() < 0.5
                if rev:
                    fr = fr[::-1]
                qs = []
                if res >= hi_res:
                    if rng.random() < 0.5:
                        qs.append(dict(type="noul", instr=rng.choice(
                            ["Is this video playing forward, as it was recorded (not reversed)?",
                             "Is time running forward in this clip (not played in reverse)?"]), target=0.0 if rev else 1.0, task="aot"))
                    else:
                        o, t = shuffle_choice(rng, ["It plays forward, as recorded.", "It plays in reverse."], 1 if rev else 0)
                        qs.append(dict(type="choice", instr="Is this clip playing forward or in reverse?", options=o,
                                       target=t, task="aot"))
                cl = cam_label(st, sa, sb)
                if cl:
                    cl = CAM_FLIP[cl] if rev else cl
                    keys = [cl] + rng.sample([k for k in CAM_OPTS if k != cl], rng.choice([3, 4, 6]))
                    o, t = shuffle_choice(rng, [CAM_OPTS[k] for k in keys], 0)
                    qs.append(dict(type="choice", instr="How does the camera move in this clip?", options=o, target=t,
                                   task="camera", cam=cl))
                    if rng.random() < 0.5:
                        qs.append(dict(type="noul", instr="Is the camera moving in this clip?",
                                       target=0.0 if cl == "static" else 1.0, task="camera_yn"))
                if not rev and split == "train":
                    qs.append(dict(type="noul", instr=rng.choice(YN_T).format(a=c["label"]), target=1.0, task="action"))
                    neg = rng.choice(labels)
                    if neg != c["label"]:
                        qs.append(dict(type="noul", instr=rng.choice(YN_T).format(a=neg), target=0.0, task="action"))
                self.emit(("calib_k700" if calib else "train_k700") if split == "train" else "eval_k700", self.rec(
                    "gv_k700", p, fr, 1.0 / fps, W, H, mv, qs, rev=rev))
            if split == "eval" or calib:
                continue
            res_clip = float(np.mean(st["res"]))
            # ---- dedicated arrow-of-time presentation at 4-8 fps (the pilot did not learn AoT from the 2-4 fps mixes)
            if res_clip >= hi_res and rng.random() < a.p_aot:
                stride = rng.choice([1, 2])
                n = rng.choice([12, 16])
                span = (n - 1) * stride + 1
                if span <= N:
                    i0 = rng.randrange(0, N - span + 1)
                    sa, sb = i0 // 2, (i0 + span - 1) // 2
                    if motion(st, sa, sb)[0] >= hi_res:
                        rev2 = rng.random() < 0.5
                        fr = list(range(i0, i0 + span, stride))
                        q = (dict(type="noul", instr=rng.choice(["Is this video playing forward, as it was recorded (not reversed)?",
                                                                  "Is time running forward in this clip (not played in reverse)?"]),
                                  target=0.0 if rev2 else 1.0, task="aot")
                             if rng.random() < 0.5 else
                             dict(zip(("type", "instr", "options", "target", "task"),
                                      ("choice", "Is this clip playing forward or in reverse?",
                                       *shuffle_choice(rng, ["It plays forward, as recorded.", "It plays in reverse."], 1 if rev2 else 0), "aot"))))
                        self.emit("train_k700", self.rec("gv_aot", p, fr[::-1] if rev2 else fr, stride / SFPS, W, H, 64, [q], rev=rev2))
            # ---- speed
            if res_clip >= hi_res and rng.random() < a.p_speed:
                lvl = rng.choice([0, 1, 1, 2, 3])
                stride = {0: 1, 1: 2, 2: 4, 3: 8}[lvl]
                n = rng.choice([8, 10, 12])
                span = (n - 1) * stride + 1
                if span <= N:
                    i0 = rng.randrange(0, N - span + 1)
                    tgt = [0.0] * 4; tgt[lvl] = 1.0
                    self.emit("train_k700", self.rec("gv_speed", p, list(range(i0, i0 + span, stride)), 0.25, W, H, mv, [
                        dict(type="score", instr="How fast is this video playing compared to real time?",
                             options=SPEED_LEVELS, target=tgt, task="speed")]))
            # ---- freeze vs moving
            if rng.random() < a.p_freeze:
                n = rng.choice([6, 8])
                if rng.random() < 0.5 or res_clip < hi_res:
                    k = rng.randrange(N)
                    fr, tgt = [k] * n, 0.0
                else:
                    i0 = rng.randrange(0, max(1, N - 2 * n))
                    fr, tgt = list(range(i0, min(N, i0 + 2 * n), 2)), 1.0
                self.emit("train_k700", self.rec("gv_freeze", p, fr, 0.5, W, H, 64, [
                    dict(type="noul", instr=rng.choice(["Does anything move in this clip?",
                                                         "Is there any motion in this clip (people, objects or camera)?"]),
                         target=tgt, task="freeze")]))
            # ---- order (cut, gap, cut)
            if res_clip >= hi_res and rng.random() < a.p_order and N >= 64:
                m = 6  # frames per part at 4 fps
                A0 = rng.randrange(0, N - 2 * 2 * m - 8)
                B0 = A0 + 2 * m + rng.randrange(8, max(9, N - A0 - 4 * m))
                if B0 + 2 * m <= N:
                    A = list(range(A0, A0 + 2 * m, 2)); B = list(range(B0, B0 + 2 * m, 2))
                    sw = rng.random() < 0.5
                    self.emit("train_k700", self.rec("gv_order", p, (B + A) if sw else (A + B), 0.25, W, H, 64, [
                        dict(type="noul", instr="This clip has a cut in the middle. Does the part after the cut happen later "
                                                "in time than the part before the cut?", target=0.0 if sw else 1.0, task="order")]))
            # ---- synthetic repetition
            if res_clip >= hi_res and rng.random() < a.p_synrep:
                m = rng.choice([3, 4, 5])
                i0 = rng.randrange(0, max(1, N - 2 * m))
                S = list(range(i0, min(N, i0 + 2 * m), 2))
                k = rng.choice([1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8])
                pingpong = rng.random() < 0.5
                cyc = (S + S[::-1][1:-1]) if pingpong else S
                fr = cyc * k
                if len(fr) > 48:
                    continue
                cands = sorted(set([k] + rng.sample([x for x in range(1, 11) if x != k], 4)))
                tgt = [1.0 if x == k else 0.0 for x in cands]
                what = "back-and-forth motion" if pingpong else "same short motion (the clip loops)"
                self.emit("train_k700", self.rec("gv_synrep", p, fr, 0.25, W, H, 48, [
                    dict(type="choice", instr=f"How many times is the {what} repeated in this clip?",
                         options=[f"{x} time" + ("s" if x > 1 else "") for x in cands], target=tgt, task="synrep", count=k)]))

    # -------------------------------------------------------------------------------------------- Countix
    def countix(self):
        rng, a = self.rng, self.a
        pools = {}
        for name in ("k700val", "countix_train"):
            f = f"{a.root}/clips/{name}/index.jsonl"
            if os.path.exists(f):
                for l in open(f):
                    c = json.loads(l)
                    b = os.path.basename(c["path"])[:-4]
                    pools[(b[:11], int(b[12:18]))] = c
        for split_name, f in (("train", "countix_train.csv"), ("val", "countix_val.csv")):
            for r in csv.DictReader(open(f"{a.root}/raw/ann/countix/{f}")):
                c = pools.get((r["video_id"], int(r["kinetics_start"])))
                if c is None:
                    continue
                if r["video_id"] in self.ex_ytid or rel(c["path"], self.root) in self.ex_paths:
                    self.n_ex["countix"] += 1
                    continue
                cnt = int(r["count"])
                s0 = float(r["repetition_start"]) - int(r["kinetics_start"])
                s1 = float(r["repetition_end"]) - int(r["kinetics_start"])
                D = s1 - s0
                if D <= 0.5 or cnt < 1:
                    continue
                fps = min(8.0, max(2.0, 4.0 * cnt / D))
                stride = max(1, int(round(SFPS / fps)))
                fps = SFPS / stride
                if fps < 2.5 * cnt / D:          # would alias
                    continue
                fr = list(range(int(s0 * SFPS), min(c["n"], int(math.ceil(s1 * SFPS))), stride))
                if len(fr) > 48:
                    stride2 = math.ceil(len(fr) / 48)
                    if fps / stride2 < 2.5 * cnt / D:
                        continue
                    fr = fr[::stride2]; stride *= stride2
                if len(fr) < 4:
                    continue
                dt = stride / SFPS
                d = sorted({cnt, cnt + 1, max(1, cnt - 1), cnt + 2, max(1, cnt - 2), cnt * 2, max(1, cnt // 2)} - {cnt})
                cands = sorted([cnt] + rng.sample(d, min(4, len(d))))
                act = r["class"]
                qs = [dict(type="choice", instr=f"How many times does the person do this repeated action ({act}) in this clip?",
                           options=[str(x) for x in cands], target=[1.0 if x == cnt else 0.0 for x in cands], task="count", count=cnt)]
                k = rng.choice([cnt, rng.choice(cands)])
                qs.append(dict(type="noul", instr=f"Is the repeated action ({act}) done exactly {k} times in this clip?",
                               target=1.0 if k == cnt else 0.0, task="count_yn"))
                tgt_split = "train_countix" if split_name == "train" else "eval_countix"
                self.emit(tgt_split, self.rec("gv_countix", c["path"], fr, dt, c["w"], c["h"], 48, qs, count=cnt))

    # -------------------------------------------------------------------------------------------- RepCount-A train
    def repcount(self):
        """windows of RepCount-A train videos; label = number of repetitions COMPLETED inside the window (cycle end in
        [t0, t1)) -> additive over non-overlapping windows, which is how eval_video.py repcount counts a whole video."""
        rng, a = self.rng, self.a
        for split in ("train", "validation"):
            f = f"{a.root}/clips/repcount_{split}/index.jsonl"
            for l in open(f):
                c = json.loads(l)
                cyc = c["cycles"]
                if not cyc or rel(c["path"], self.root) in self.ex_paths:
                    continue
                N, dur = c["n"], c["n"] / SFPS
                act = c["action"].replace("_", " ")
                for _ in range(a.rc_windows):
                    L, stride = 4.0, 1           # same as eval: 4 s at 8 fps = 32 frames
                    if rng.random() < 0.3:
                        L, stride = 8.0, 2
                    t0 = rng.uniform(0, max(0.0, dur - L))
                    fr = list(range(int(t0 * SFPS), min(N, int((t0 + L) * SFPS)), stride))
                    if len(fr) < 8:
                        continue
                    k = sum(1 for s, e in cyc if t0 <= e < t0 + L)
                    if k > 12:
                        continue
                    qs = []
                    instr = f"How many repetitions of this repeated action ({act}) are completed in this clip?"
                    if rng.random() < 0.5:
                        qs.append(dict(type="choice", instr=instr, options=[str(i) for i in range(13)],
                                       target=[1.0 if i == k else 0.0 for i in range(13)], task="rep_window", count=k))
                    else:
                        d = sorted({max(0, k - 2), max(0, k - 1), k + 1, k + 2, k + 3} - {k})
                        cands = sorted([k] + rng.sample(d, min(4, len(d))))
                        qs.append(dict(type="choice", instr=instr, options=[str(x) for x in cands],
                                       target=[1.0 if x == k else 0.0 for x in cands], task="rep_window", count=k))
                    qs.append(dict(type="noul", instr=f"Does the person finish at least one full repetition of {act} in this clip?",
                                   target=1.0 if k >= 1 else 0.0, task="rep_any"))
                    self.emit("train_repcount" if split == "train" else "calib_repcount",
                              self.rec("gv_repcount", c["path"], fr, stride / SFPS, c["w"], c["h"], 48, qs, count=k))

    # -------------------------------------------------------------------------------------------- Charades
    def charades(self):
        rng, a = self.rng, self.a
        ann = f"{a.root}/raw/ann/Charades"
        cls = dict(l.strip().split(" ", 1) for l in open(f"{ann}/Charades_v1_classes.txt"))
        obj_of = {}
        for l in open(f"{ann}/Charades_v1_mapping.txt"):
            c, o, v = l.split()
            obj_of[c] = o
        idx = {}
        f = f"{a.root}/clips/charades/index.jsonl"
        for l in open(f):
            c = json.loads(l); idx[c["vid"]] = c
        sta = defaultdict(list)
        for sp in ("train", "test"):
            for l in open(f"{a.root}/raw/ann/charades_sta_{sp}.txt"):
                head, s = l.strip().split("##")
                v, t0, t1 = head.split()
                sta[v].append((float(t0), float(t1), s.strip().rstrip(".")))
        for sp in ("train", "test"):
            for r in csv.DictReader(open(f"{ann}/Charades_v1_{sp}.csv")):
                v = r["id"]
                c = idx.get(v)
                if c is None:
                    continue
                if v in self.ex_charades or rel(c["path"], self.root) in self.ex_paths:
                    self.n_ex["charades"] += 1
                    continue
                acts = []
                for x in (r["actions"] or "").split(";"):
                    if x:
                        k, s0, s1 = x.split()
                        acts.append((k, float(s0), float(s1)))
                if not acts:
                    continue
                self._charades_video(sp, v, c, acts, sta.get(v, []), cls, obj_of)

    def _charades_video(self, sp, v, c, acts, sta, cls, obj_of):
        rng, a = self.rng, self.a
        N, W, H = c["n"], c["w"], c["h"]
        dur = N / SFPS
        present = {k for k, _, _ in acts}
        n_win = a.ch_windows if sp == "train" else 2
        # dedicated order records: two annotated actions in sequence, window just covering both (<= 12 s at 2 fps)
        pairs = [(x, y) for x in acts for y in acts if x[0] != y[0] and x[2] <= y[1] + 0.5 and y[1] - x[2] < 4.0
                 and y[2] - x[1] <= 12.0 and x[2] - x[1] >= 1.0 and y[2] - y[1] >= 1.0 and x[0] not in (y[0], REV.get(y[0]))]
        for x, y in rng.sample(pairs, min(2, len(pairs))):
            t0, t1 = max(0.0, x[1] - 0.5), min(dur, y[2] + 0.5)
            fr = list(range(int(t0 * SFPS), min(N, int(t1 * SFPS)), 4))[:24]
            if len(fr) < 4:
                continue
            rev = rng.random() < 0.3
            first, second = (y, x) if rev else (x, y)
            o, t = shuffle_choice(rng, [phrase(cls[first[0]]), phrase(cls[second[0]])], 0)
            qs = [dict(type="choice", instr="Which of these happens first in this clip?",
                       options=[s[0].upper() + s[1:] + "." for s in o], target=t, task="event_order")]
            if not rev:
                k = rng.random() < 0.5
                a1, a2 = (x, y) if k else (y, x)
                qs.append(dict(type="noul", instr=f"Does the person start {phrase(cls[a1[0]])} before {phrase(cls[a2[0]])}?",
                               target=1.0 if k else 0.0, task="event_order_yn"))
            split = ("calib_charades" if h20(v) == 1 else "train_charades") if sp == "train" else "eval_charades"
            self.emit(split, self.rec("gv_charades_order", c["path"], fr[::-1] if rev else fr, 0.5, W, H, 64, qs, rev=rev, vid=v))
        for _ in range(n_win):
            fps = rng.choice([2, 2, 4])
            L = rng.choice([2, 3, 4, 6, 8]) if fps == 2 else rng.choice([2, 3, 4])
            if rng.random() < 0.4 and acts:      # center some windows on an action (more positives)
                k, s0, s1 = rng.choice(acts)
                t0 = min(max(0.0, rng.uniform(s0 - L / 2, max(s0 - L / 2, s1 - L / 2))), max(0.0, dur - L))
            else:
                t0 = rng.uniform(0, max(0.0, dur - L))
            t1 = t0 + L
            stride = SFPS // fps
            fr = list(range(int(t0 * SFPS), min(N, int(t1 * SFPS)), stride))[:16]
            if len(fr) < 4:
                continue
            rev = rng.random() < 0.25
            pos, temporal_neg, unsure = [], [], set()
            for k, s0, s1 in acts:
                ov = max(0.0, min(t1, s1) - max(t0, s0))
                cover = ov / max(1e-6, min(L, s1 - s0))
                if cover >= 0.6:
                    pos.append(k)
                elif ov > 0:
                    unsure.add(k)
            for k, s0, s1 in acts:
                if k not in pos and k not in unsure:
                    temporal_neg.append(k)
            pos = sorted(set(pos)); temporal_neg = sorted(set(temporal_neg) - set(pos))
            if rev:   # reversed: reversible actions flip, non-reversible positives are dropped (stay unsure)
                pos2 = []
                for k in pos:
                    if k in REV:
                        pos2.append(REV[k])
                pos = pos2
                temporal_neg = [k for k in temporal_neg if k not in REV and k not in pos]
            blocked = set(pos) | unsure | ({REV[k] for k in unsure if k in REV})
            qs = []
            for k in rng.sample(pos, min(3, len(pos))):
                qs.append(dict(type="noul", instr=rng.choice(YN_T).format(a=phrase(cls[k])), target=1.0, task="event"))
                if k in REV and REV[k] not in blocked:  # the reversal antonym is the hardest negative
                    qs.append(dict(type="noul", instr=rng.choice(YN_T).format(a=phrase(cls[REV[k]])), target=0.0,
                                   task="event_rev_neg"))
            for k in rng.sample(temporal_neg, min(2, len(temporal_neg))):
                if k not in blocked:
                    qs.append(dict(type="noul", instr=rng.choice(YN_T).format(a=phrase(cls[k])), target=0.0, task="event_tneg"))
            same_obj = [k for k in cls if k not in present and k not in blocked and any(obj_of.get(k) == obj_of.get(p) for p in pos)]
            for k in rng.sample(same_obj, min(1, len(same_obj))):
                qs.append(dict(type="noul", instr=rng.choice(YN_T).format(a=phrase(cls[k])), target=0.0, task="event_objneg"))
            others = [k for k in cls if k not in present and k not in blocked]
            k = rng.choice(others)
            qs.append(dict(type="noul", instr=rng.choice(YN_T).format(a=phrase(cls[k])), target=0.0, task="event_neg"))
            # choice: one positive vs hard negatives
            if pos:
                k = rng.choice(pos)
                negs = [x for x in ([REV.get(k)] + temporal_neg + same_obj) if x and x not in blocked]
                negs = list(dict.fromkeys(negs))[:3]
                negs += rng.sample([x for x in others if x not in negs], 4 - 1 - len(negs))
                o, t = shuffle_choice(rng, [phrase(cls[x]) for x in [k] + negs], 0)
                qs.append(dict(type="choice", instr="Which of these does the person do in this clip?",
                               options=[x[0].upper() + x[1:] + "." for x in o], target=t, task="event_choice"))
            # order of two actions fully inside the window (forward or reversed)
            inside = [(k, s0, s1) for k, s0, s1 in acts if s0 >= t0 and s1 <= t1 and s1 - s0 >= 0.5]
            pairs = [(x, y) for x in inside for y in inside if x[2] <= y[1] and x[0] != y[0]]
            if pairs and not rev:
                x, y = rng.choice(pairs)
                o, t = shuffle_choice(rng, [phrase(cls[x[0]]), phrase(cls[y[0]])], 0)
                qs.append(dict(type="choice", instr="Which of these happens first in this clip?",
                               options=[s[0].upper() + s[1:] + "." for s in o], target=t, task="event_order"))
            # STA sentences (forward only)
            if not rev:
                for s0, s1, s in sta:
                    ov = max(0.0, min(t1, s1) - max(t0, s0))
                    cover = ov / max(1e-6, min(L, s1 - s0))
                    if cover >= 0.6 or ov == 0:
                        if cover >= 0.6 or rng.random() < 0.5:
                            qs.append(dict(type="noul", instr=rng.choice(STA_T).format(s=s), target=1.0 if cover >= 0.6 else 0.0,
                                           task="sta"))
            if rev:
                fr = fr[::-1]
            split = ("calib_charades" if h20(v) == 1 else "train_charades") if sp == "train" else "eval_charades"
            self.emit(split, self.rec("gv_charades", c["path"], fr, 1.0 / fps, W, H, rng.choice([64, 96]), qs[:10], rev=rev,
                                      vid=v, t0=round(t0, 2), t1=round(t1, 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/gv"))
    ap.add_argument("--out", default=os.path.expanduser("~/gv/records"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k700_pres", type=int, default=2)
    ap.add_argument("--p_speed", type=float, default=0.5)
    ap.add_argument("--p_freeze", type=float, default=0.15)
    ap.add_argument("--p_order", type=float, default=0.3)
    ap.add_argument("--p_synrep", type=float, default=0.4)
    ap.add_argument("--p_aot", type=float, default=1.0)
    ap.add_argument("--ch_windows", type=int, default=8)
    ap.add_argument("--rc_windows", type=int, default=12)
    ap.add_argument("--only", default="k700,countix,repcount,charades")
    a = ap.parse_args()
    g = Gen(a)
    for name in a.only.split(","):
        try:
            getattr(g, name)()
        except FileNotFoundError as e:
            print("skip", name, e, flush=True)
    os.makedirs(a.out, exist_ok=True)
    summary = {}
    for split, recs in g.out.items():
        g.rng.shuffle(recs)
        with open(f"{a.out}/{split}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        tasks = defaultdict(int)
        for r in recs:
            for q in r["questions"]:
                tasks[q.get("task", "?")] += 1
        summary[split] = dict(records=len(recs), questions=sum(tasks.values()), tasks=dict(sorted(tasks.items())),
                              plen_mean=round(float(np.mean([r["plen"] for r in recs])), 1))
    summary["excluded"] = dict(g.n_ex)
    json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
