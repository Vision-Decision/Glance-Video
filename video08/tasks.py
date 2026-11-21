"""Question / target construction shared by S1 training and evaluation (heads and the generative baseline see the SAME
token ids). Every builder takes the video row, the query time t (seconds into the clip) and an rng."""
import random

ASSIST = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Q:
    def __init__(self, tok, classes, mapping):
        self.tok, self.classes, self.mapping = tok, classes, mapping
        self.C = sorted(classes)

    def phrase(self, c):
        p = self.classes[c]; return p[0].lower() + p[1:]

    def ids(self, text):
        return self.tok.encode(f"<|vision_end|>{text}{ASSIST}", add_special_tokens=False)

    def choice_ids(self, stem, options):
        """-> ids, option end indices (last token of each option text)"""
        e = lambda s: self.tok.encode(s, add_special_tokens=False)
        ids = e(f"<|vision_end|>{stem}"); ends = []
        for i, o in enumerate(options):
            ids += e(f" ({'ABCD'[i]}) {o}"); ends.append(len(ids) - 1)
        ids += e(f". Answer with the letter.{ASSIST}")
        return ids, ends

    # ------------------------------------------------------------ presence
    def presence(self, row, t, rng, n=8):
        """'right now' and 'so far' questions with labels. Negatives: present-but-not-now / not-yet (temporal hard),
        absent sharing an object or verb (semantic hard), absent random."""
        acts = row["actions"]; present = {c for c, _, _ in acts}
        now = {c for c, s, e in acts if s <= t <= e}
        sofar = {c for c, s, e in acts if s <= t}
        absent = [c for c in self.C if c not in present]
        objs = {self.mapping[c][0] for c in present}; verbs = {self.mapping[c][1] for c in present}
        hard = [c for c in absent if self.mapping[c][0] in objs or self.mapping[c][1] in verbs]
        out = []
        for kind, pos, tneg in (("now", now, present - now), ("sofar", sofar, present - sofar)):
            P = rng.sample(sorted(pos), min(len(pos), n // 4))
            T = rng.sample(sorted(tneg), min(len(tneg), n // 4))
            H = rng.sample(hard, min(len(hard), n // 4))
            R = rng.sample([c for c in absent if c not in H], n - len(P) - len(T) - len(H))
            for c in P + T + H + R:
                q = (f"Is someone {self.phrase(c)} right now? Answer yes or no." if kind == "now" else
                     f"Has someone been {self.phrase(c)} at any point so far? Answer yes or no.")
                out.append(dict(task="presence", kind=kind, cls=c, y=int(c in pos), ids=self.ids(q),
                                neg=("pos" if c in pos else "temporal" if c in T else "hard" if c in H else "random")))
        return out

    # ------------------------------------------------------------ temporal (segments so far, normalised by t)
    def temporal(self, row, t, rng, n=3):
        out = []
        by = {}
        for c, s, e in row["actions"]:
            if s < t:
                by.setdefault(c, []).append((s / t, min(e, t) / t))
        cs = rng.sample(sorted(by), min(len(by), n))
        for c in cs:
            q = f"When has someone been {self.phrase(c)} so far? Give the start and end times in seconds."
            out.append(dict(task="temporal", cls=c, segs=sorted(by[c]), ids=self.ids(q), t=t))
        sta = [x for x in row.get("sta", []) if x[1] < t]
        for s, a, b in rng.sample(sta, min(len(sta), 2)):
            q = f"Give the query: '{s}', when does the described content occur in the video? Please return the start and end timestamps in seconds."
            out.append(dict(task="temporal", cls="sta", segs=[(a / t, min(b, t) / t)], ids=self.ids(q), t=t, text=s))
        absent = [c for c in self.C if c not in {x[0] for x in row["actions"]}]
        c = rng.choice(absent)
        out.append(dict(task="temporal", cls=c, segs=[], t=t,
                        ids=self.ids(f"When has someone been {self.phrase(c)} so far? Give the start and end times in seconds.")))
        return out

    # ------------------------------------------------------------ which one
    def choice(self, row, t, rng):
        now = sorted({c for c, s, e in row["actions"] if s <= t <= e})
        if not now:
            return []
        c = rng.choice(now); absent = [x for x in self.C if x not in {a[0] for a in row["actions"]}]
        opts = rng.sample(absent, 3) + [c]; rng.shuffle(opts)
        ids, ends = self.choice_ids("Which of these is the person doing right now?", [self.classes[o] for o in opts])
        return [dict(task="choice", y=opts.index(c), ids=ids, ends=ends, opts=opts)]

    # ------------------------------------------------------------ counting
    def count(self, row, t, rng):
        k = sum(1 for s, e in row["reps"] if e <= t + 1e-6)
        act = row["action"].replace("_", " ")
        return [dict(task="count", y=k, ids=self.ids(f"How many repetitions of '{act}' has the person completed so far? "
                                                     f"Answer with a single number."))]
