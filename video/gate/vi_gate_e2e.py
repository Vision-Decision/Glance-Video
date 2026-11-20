"""Video-Index offline gate, part 2: end to end through the agent path (gen-host).

The same docker image, network, gateway and `vdmscan` CLI as the +vdmscan agents; 2 CPUs; NON-Video-Index videos only.

  build  (host)       python3 vi_gate_e2e.py build --root ~/videoindex/gate
      - Charades-STA test HOLD videos (video/gate/vi_gate.py split) as 2-fps mp4 from the pilot frames (same frames the pilot
        scored), synthetic long videos for counting (one positive video + its clean negative videos, shuffled), and two
        long 720p30 videos (10 min, 120 min) for latency; jobs.json lists every call to make.
  run    (container)  python3 vi_gate_e2e.py run --root /gate     -> calls.jsonl (one line per vdmscan call + output)
  report (host)       python3 vi_gate_e2e.py report --root ~/videoindex/gate  -> gate_e2e.json
"""
import argparse, glob, gzip, hashlib, json, math, os, random, subprocess, sys, time


def split_of(v):
    return "CAL" if int(hashlib.sha1(v.encode()).hexdigest(), 16) % 2 == 0 else "HOLD"


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def build(a):
    R = a.root
    meta = json.load(open(f"{R}/sta/charades_pilot2/meta.json"))
    pil = json.load(gzip.open(f"{R}/results/gv0/scan/pilot_clips_shortgv.json.gz", "rt"))
    S = pil["scans"]
    qvid = {s["qid"]: s["vid"] for s in S if s["pos"]}
    ok = lambda s: split_of(s["vid"]) == "HOLD" and split_of(qvid.get(s["qid"], s["vid"])) == "HOLD"
    hold = [s for s in S if ok(s)]
    rng = random.Random(0)
    os.makedirs(f"{R}/mp4", exist_ok=True)

    def mp4(v):
        out = f"{R}/mp4/{v}.mp4"
        if not os.path.exists(out):
            sh(f"ffmpeg -v error -y -framerate 2 -i {R}/sta/charades_pilot2/frames/{v}/%03d.jpg -c:v libx264 -crf 18 -pix_fmt yuv420p "
               f"-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' {out}")
        return out
    jobs = []
    # A: per-video parity / ever / when (positive + up to 2 negative queries per video)
    byv = {}
    for s in hold:
        byv.setdefault(s["vid"], []).append(s)
    vids = sorted(v for v in byv if any(x["pos"] for x in byv[v]))
    rng.shuffle(vids)
    for v in vids[: a.n_videos]:
        mp4(v)
        P = [x for x in byv[v] if x["pos"]][:1]
        N = [x for x in byv[v] if not x["pos"]][:2]
        for x in P + N:
            jobs.append(dict(kind="A", video=f"mp4/{v}.mp4", vid=v, op="curve", query=x["text"], pos=x["pos"], qid=x["qid"],
                             labels=x["labels"], pilot=x["scores"]["g_short"]))
    # B: synthetic long videos for counting: own positive video + its clean negatives (same query), shuffled
    negs = {}
    for s in hold:
        if not s["pos"]:
            negs.setdefault(s["qid"], []).append(s["vid"])
    pos = [s for s in hold if s["pos"] and any(s["labels"]) and len(negs.get(s["qid"], [])) >= 4]
    rng.shuffle(pos)
    os.makedirs(f"{R}/long", exist_ok=True)
    for s in pos[: a.n_count]:
        parts = [s["vid"]] + negs[s["qid"]][:8]
        rng.shuffle(parts)
        out = f"{R}/long/cnt_{s['qid']}.mp4"
        if not os.path.exists(out):
            lst = f"{R}/long/cnt_{s['qid']}.txt"
            open(lst, "w").write("".join(f"file '{mp4(p)}'\n" for p in parts))
            sh(f"ffmpeg -v error -y -f concat -safe 0 -i {lst} -vf 'scale=480:270:force_original_aspect_ratio=decrease,pad=480:270:(ow-iw)/2:(oh-ih)/2,setsar=1' "
               f"-r 2 -c:v libx264 -crf 18 -pix_fmt yuv420p {out}")
        inst = [i for i in meta["queries"] if i["qid"] == s["qid"]][0]["instances"]
        merged = []
        for i0, i1 in sorted(inst):
            if merged and i0 <= merged[-1][1] + 1.0:
                merged[-1][1] = max(merged[-1][1], i1)
            else:
                merged.append([i0, i1])
        for op in ("count", "when"):
            jobs.append(dict(kind="B", video=f"long/cnt_{s['qid']}.mp4", op=op, query=s["text"], gt_count=len(merged), qid=s["qid"],
                             n_parts=len(parts), pos_part=parts.index(s["vid"])))
    # C: long 720p30 videos (latency): concatenations of HOLD videos, upscaled; a few queries each
    allv = sorted({s["vid"] for s in hold})
    for mins in (10, 120):
        out = f"{R}/long/lat_{mins}min.mp4"
        if not os.path.exists(out):
            lst = f"{R}/long/lat_{mins}.txt"
            sel, tot = [], 0.0
            i = 0
            while tot < mins * 60:
                v = allv[i % len(allv)]; i += 1
                sel.append(v); tot += meta["videos"][v]["duration"]
            open(lst, "w").write("".join(f"file '{mp4(p)}'\n" for p in sel))
            sh(f"ffmpeg -v error -y -f concat -safe 0 -i {lst} -vf 'scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30' "
               f"-c:v libx264 -preset veryfast -crf 23 -g 60 -pix_fmt yuv420p -t {mins * 60} {out}")
        for k, q in enumerate(["a person opens a door", "a person is holding a cup", "a person sits down on a chair", "a person turns on a light"]):
            jobs.append(dict(kind="C", video=f"long/lat_{mins}min.mp4", op="when", query=q, minutes=mins, k=k))
        jobs.append(dict(kind="C", video=f"long/lat_{mins}min.mp4", op="when", query="a person opens a door", minutes=mins, k=4,
                         inn=f"{mins * 60 // 2}-{mins * 60 // 2 + 120}"))
    # D: RepCount-A test (152): reps (window head) and count (event runs) over the whole clip
    for l in open(os.path.expanduser("~/gv/clips/repcount_test/index.jsonl")):
        c = json.loads(l)
        act = c["action"].replace("_", " ")
        jobs.append(dict(kind="D", video=f"repcount/{os.path.basename(c['path'])}", op="reps", query=act, gt_count=c["count"], dur=c["dur"]))
        jobs.append(dict(kind="D", video=f"repcount/{os.path.basename(c['path'])}", op="count", query=f"the person completes one {act}",
                         gt_count=c["count"], dur=c["dur"]))
    json.dump(jobs, open(f"{R}/jobs.json", "w"))
    print({k: sum(j["kind"] == k for j in jobs) for k in "ABCD"})


def run(a):
    R = a.root
    jobs = json.load(open(f"{R}/jobs.json"))
    if a.kinds:
        jobs = [j for j in jobs if j["kind"] in a.kinds]
    done = set()
    outp = f"{a.out}/calls.jsonl"
    if os.path.exists(outp):
        done = {json.loads(l)["i"] for l in open(outp)}
    os.environ["VDMSCAN_LOG"] = f"{a.out}/vdmscan_log.jsonl"
    for i, j in enumerate(jobs):
        if i in done:
            continue
        cmd = ["vdmscan", j["op"], f"{R}/{j['video']}", j["query"], "--json"] + (["--in", j["inn"]] if j.get("inn") else [])
        if j["kind"] == "D" and j["op"] == "reps" and j["dur"] > 180:
            cmd += ["--in", "0-180"]
        t = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.time() - t
        res = None
        for line in p.stdout.splitlines():
            if line.startswith("{"):
                try:
                    res = json.loads(line)
                except Exception:
                    pass
        summ = [l for l in p.stdout.splitlines() if l.startswith("#")]
        rec = dict(i=i, kind=j["kind"], op=j["op"], wall=round(wall, 2), summary=summ[-2:], rc=p.returncode, err=p.stderr[-300:])
        if res:
            for k in ("n_clips", "sec", "count", "count_argmax", "ever", "when", "curve", "windows"):
                if k in res:
                    rec[k] = res[k]
        with open(outp, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(i, j["kind"], j["op"], f"{wall:.1f}s", summ[-1][:120] if summ else p.stdout[-200:], flush=True)


def report(a):
    import numpy as np
    R = a.root
    jobs = json.load(open(f"{R}/jobs.json"))
    calls = {json.loads(l)["i"]: json.loads(l) for l in open(f"{R}/out/calls.jsonl")}
    vlog = [json.loads(l) for l in open(f"{R}/out/vdmscan_log.jsonl")]
    T = json.load(open(f"{R}/CALIB.json"))["T"]
    out = {}
    # A
    A = [(j, calls[i]) for i, j in enumerate(jobs) if j["kind"] == "A" and i in calls and "curve" in calls[i]]
    rs, agree, fa, det = [], [], [], []
    for j, c in A:
        p = np.clip(np.array([x["p"] for x in c["curve"]]), 1e-6, 1 - 1e-6)
        s = T * np.log(p / (1 - p))
        q = np.array(j["pilot"])
        n = min(len(s), len(q))
        if n > 2:
            rs.append(np.corrcoef(s[:n], q[:n])[0, 1])
            agree.append(float(((s[:n] > 0) == (q[:n] > 0)).mean()))
        (det if j["pos"] else fa).append(bool(c["ever"]["ever"]))
    out["A_parity"] = dict(n_calls=len(A), logit_corr_median=float(np.median(rs)), decision_agree=float(np.mean(agree)),
                           ever_false_alarm_neg=float(np.mean(fa)), n_neg=len(fa), ever_detection_pos=float(np.mean(det)), n_pos=len(det))
    # B
    B = [(j, calls[i]) for i, j in enumerate(jobs) if j["kind"] == "B" and j["op"] == "count" and i in calls and "count" in calls[i]]
    gt = np.array([j["gt_count"] for j, _ in B]); pr = np.array([c["count"]["count"] for _, c in B])
    out["B_count_synthetic"] = dict(n=len(B), gt_mean=float(gt.mean()), exact=float((gt == pr).mean()), within1=float((abs(gt - pr) <= 1).mean()),
                                    mae=float(abs(gt - pr).mean()), pred_hist={str(k): int((pr == k).sum()) for k in range(6)})
    # D
    for op in ("reps", "count"):
        D = [(j, calls[i]) for i, j in enumerate(jobs) if j["kind"] == "D" and j["op"] == op and i in calls and "count" in calls[i]]
        g = np.array([j["gt_count"] for j, _ in D], float)
        for key in (("count", "count_argmax") if op == "reps" else ("count",)):
            pr = np.array([(c[key] if not isinstance(c[key], dict) else c[key]["count"]) for _, c in D], float)
            out[f"D_repcount_{op}_{key}"] = dict(n=len(D), MAE=float(np.mean(abs(pr - g) / np.maximum(g, 1))),
                                                OBO=float(np.mean(abs(np.round(pr) - g) <= 1)))
    # latency per call (from the client log: wall time inside the container, incl. extraction / upload / queueing)
    lat = {}
    for c, j in ((calls[i], jobs[i]) for i in calls):
        lat.setdefault((j["kind"], c["op"]), []).append(c["wall"])
    first = [x for x in vlog if x.get("encoded")]
    follow = [x for x in vlog if x.get("encoded") is False]
    out["latency"] = {
        "first_call_per_video(encode)": dict(n=len(first), median_s=float(np.median([x["sec_total"] for x in first])),
                                             p90_s=float(np.percentile([x["sec_total"] for x in first], 90))),
        "follow_up_call(cached)": dict(n=len(follow), median_s=float(np.median([x["sec_total"] for x in follow])),
                                       p90_s=float(np.percentile([x["sec_total"] for x in follow], 90)),
                                       server_median_s=float(np.median([x["server_sec"] for x in follow]))),
        "by_kind_op": {f"{k[0]}:{k[1]}": dict(n=len(v), median_s=float(np.median(v)), max_s=float(np.max(v))) for k, v in sorted(lat.items())}}
    for mins in (10, 120):
        L = [(j, calls[i]) for i, j in enumerate(jobs) if j["kind"] == "C" and j.get("minutes") == mins and i in calls]
        out["latency"][f"C_{mins}min"] = [dict(k=j["k"], inn=j.get("inn"), wall=c["wall"], n_clips=c.get("n_clips"), summary=c["summary"][-1:]) for j, c in L]
    json.dump(out, open(f"{R}/gate_e2e.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run", "report"])
    ap.add_argument("--root", default=os.path.expanduser("~/videoindex/gate"))
    ap.add_argument("--out", default="/gate/out")
    ap.add_argument("--kinds", default="")
    ap.add_argument("--n_videos", type=int, default=40)
    ap.add_argument("--n_count", type=int, default=30)
    a = ap.parse_args()
    dict(build=build, run=run, report=report)[a.cmd](a)
