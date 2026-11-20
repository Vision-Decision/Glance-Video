"""Glance-Video decontamination list against Video-Index v1.0 (840 items) + v1.1 (144 items).

Four layers (all applied by gen_records.py):
  1. dataset level  - the 76 source benchmarks (train AND test splits) and the pools that are themselves a source or
                      are mostly reused by sources: never used (DATASET_EXCLUDED below).
  2. id level       - exact original ids: v1.1 `source_video` (repo::archive::member -> dataset + id), VATEX
                      val/test YouTube ids (Kinetics-600 clips reused by VITATECS / Vinoground / VideoEspresso).
  3. content hash   - SHA-256 of every raw source file we use vs the Video-Index video ids (sha256 of their files).
  4. perceptual     - 64-bit DCT pHash of 1-fps frames of every normalised clip vs all VI frames (2 fps for
                      videos <= 10 min); a clip is excluded when >= 2 frames (Hamming <= 6) match one VI video
                      (prep_clips.py decon). The whole source video of a matched clip is excluded.

  python decon_list.py sha  --root ~/gv --name k700val      # sha256 of raw files -> decon/sha_<name>.jsonl
  python decon_list.py build --root ~/gv                     # -> decon/exclusions.json + decon/report.md
"""
import argparse, glob, hashlib, json, os, re
from collections import Counter, defaultdict

VI_ITEMS = ["~/videoindex/data/v1_0/items/test.jsonl", "~/videoindex/data/v1_1/items/test.jsonl"]

# pools never used for training (reason); see the source map in the report
DATASET_EXCLUDED = {
    "NExT-QA (VidOR)": "is a VI source; also reused by AoTBench, E-VQA, V2P-Bench, VCR-Bench",
    "Perception Test": "is a VI source (19 items); reused by TOC-Bench, TOMATO, TVBench, AoTBench, E.T. Bench, MVBench, ...",
    "STAR / Action Genome / AGQA": "Charades-based QA sets reused by MVBench, TVBench, E.T. Bench, E-VQA, TaskMeAnything",
    "Ego4D / EgoSchema / HourVideo": "VI sources (EgoSchema, HourVideo) and used by MLVU, E.T. Bench, MMWorld, OVO-S, ...",
    "MSR-VTT": "test split used by VITATECS (24 items), AoTBench, V2P-Bench",
    "VATEX": "val/test used by VITATECS, Vinoground, VideoEspresso (their YouTube ids are also removed from Kinetics)",
    "ActivityNet / ActivityNet Captions": "used by ReXTime, TemporalBench, VKnowU, SITE, V2P, VCR, MLVU (not used in v0; "
                                           "would need YouTube-id exclusion)",
    "YouCook2, MVBench sources (CLEVRER, NTU, FunQA, MoVQA, PAXION)": "used by E.T. Bench, VideoEspresso, MVBench",
    "all 76 Video-Index source benchmarks": "train and test splits",
}
POOLS_USED = {
    "Kinetics-700-2020 val": "id + content + pHash exclusion; VATEX val/test ytids removed (VI uses Kinetics via VATEX and DSI-Bench)",
    "Charades (train split only)": "id + content + pHash exclusion (FAVOR-Bench, TOC-Bench, TemporalBench, E.T. Bench, "
                                   "MVBench use Charades / Charades-STA clips); Charades test split held out for eval",
    "Countix (on Kinetics-700 val)": "same exclusion as Kinetics",
}


def parse_source(sv):
    """v1.1 source_video 'repo::archive::member' -> (dataset, id) for the pools we use, else (repo, member)."""
    repo, _, member = (sv.split("::") + ["", ""])[:3]
    base = os.path.basename(member).rsplit(".", 1)[0]
    b2 = base[len("charades_"):] if base.startswith("charades_") else base
    if "charades" in repo.lower() or "Charades" in member or base.startswith("charades_") or \
            re.fullmatch(r"[0-9A-Z]{5}", b2):          # Charades ids: 5 chars [0-9A-Z] (FAVOR-Bench, TOC-Bench reuse them)
        return "charades", b2[:5]
    m = re.search(r"([A-Za-z0-9_-]{11})_\d{6}_\d{6}", member)
    if m:
        return "kinetics_ytid", m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", base):
        return "youtube", base
    return repo, member


def cmd_sha(a):
    idx = [json.loads(l) for l in open(f"{a.root}/clips/{a.name}/index.jsonl")]
    srcs = sorted(set(c.get("src_path") or c["path"] for c in idx))
    raw = sorted(glob.glob(f"{a.root}/raw/{a.raw or a.name}/**/*.mp4", recursive=True))
    out = f"{a.root}/decon/sha_{a.name}.jsonl"
    done = {json.loads(l)["path"] for l in open(out)} if os.path.exists(out) else set()
    with open(out, "a") as f:
        for p in raw:
            if p in done:
                continue
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for b in iter(lambda: fh.read(1 << 20), b""):
                    h.update(b)
            f.write(json.dumps(dict(path=p, sha256=h.hexdigest())) + "\n")
    print(len(raw), "raw files hashed;", len(srcs), "clips indexed")


BENCH_CHARADES = {  # VI source benchmarks that reuse Charades videos: every Charades id they contain is excluded
    "FAVOR-Bench/FAVOR": None, "anonymous-video-benchmark/toc_bench": None,
    "OpenGVLab/MVBench": r"json/.*\.json$", "FunAILab/TVBench": r"json/.*\.json$",
    "microsoft/TemporalBench": r".*\.json$", "PolyU-ChenLab/ETBench": r"annotations/.*\.json$",
}


def cmd_bench_ids(a):
    """Charades ids used anywhere in the VI source benchmarks above (file names + annotation JSONs) -> decon/bench_charades.json"""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    out = {}
    for repo, pat in BENCH_CHARADES.items():
        ids = set()
        try:
            fs = api.list_repo_files(repo, repo_type="dataset")
        except Exception as e:
            print(repo, "unavailable", type(e).__name__); out[repo] = None; continue
        for f in fs:
            m = re.search(r"(?:^|[/_])([0-9A-Z]{5})\.mp4$", f)
            if m:
                ids.add(m.group(1))
        if pat:
            for f in fs:
                if re.match(pat, f):
                    try:   # gated repos (e.g. TemporalBench) are NOT requested: we never accept terms; pHash covers them
                        txt = open(hf_hub_download(repo, f, repo_type="dataset")).read()
                    except Exception as e:
                        print(repo, f, "unavailable", type(e).__name__); out.setdefault("_gated", []).append(repo); break
                    ids |= set(re.findall(r"(?<![0-9A-Za-z])([0-9A-Z]{5})\.mp4", txt))
                    ids |= set(re.findall(r"Charades/([0-9A-Z]{5})_", txt))          # TemporalBench: Charades/ID_start_..
                    if repo == "microsoft/TemporalBench":                             # YouTube ids (ActivityNet, COIN, FineGym)
                        out.setdefault("_tb_youtube", set()).update(
                            re.findall(r"(?:ActivityNet|COIN|FineGym)/([A-Za-z0-9_-]{11})_", txt))
        out[repo] = sorted(ids)
        print(repo, len(ids), flush=True)
    out = {k: (sorted(v) if isinstance(v, set) else v) for k, v in out.items()}
    json.dump(out, open(f"{a.root}/decon/bench_charades.json", "w"), indent=0)


def cmd_build(a):
    items = []
    for p, repo in zip(VI_ITEMS, ["Video-Index/Video-Index", "Video-Index/Video-Index-v1-1"]):
        p = os.path.expanduser(p)
        if not os.path.exists(p):                      # on a Colab VM: the official item files (metadata only)
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(repo, "items/test.jsonl", repo_type="dataset")
        items += [json.loads(l) for l in open(p)]
    benches = Counter(r["benchmark"] for r in items[:840])
    vi_sha = {r["video_id"] for r in items} | {r.get("video_v1_0") for r in items if r.get("video_v1_0")}
    ids = defaultdict(set)
    other = []
    for r in items:
        if r.get("source_video"):
            k, v = parse_source(r["source_video"])
            if k in ("charades", "kinetics_ytid", "youtube"):
                ids[k].add(v)
            else:
                other.append(r["source_video"])
    vatex = set()
    for f in glob.glob(f"{a.root}/decon/ext/vatex_*.json"):
        for d in json.load(open(f)):
            vatex.add(d["videoID"][:11])
    ids["kinetics_ytid"] |= vatex
    bc = f"{a.root}/decon/bench_charades.json"
    bench_ch = json.load(open(bc)) if os.path.exists(bc) else {}
    for k, v in bench_ch.items():
        if v and not k.startswith("_"):
            ids["charades"] |= set(v)
    ids["youtube"] |= set(bench_ch.get("_tb_youtube", []))
    sha_hits = []
    for f in glob.glob(f"{a.root}/decon/sha_*.jsonl"):
        for l in open(f):
            d = json.loads(l)
            if d["sha256"] in vi_sha:
                sha_hits.append(d["path"])
    ph = dict(n_clips=0, n_contaminated=0, contaminated=[], by_set={})
    for pf in sorted(glob.glob(f"{a.root}/decon/clip_exclusions_*.json")):
        d = json.load(open(pf))
        ph["n_clips"] += d["n_clips"]; ph["n_contaminated"] += d["n_contaminated"]; ph["contaminated"] += d["contaminated"]
        ph["by_set"][d.get("against", "vi_hashes.npz") + ":" + "_".join(d.get("names", []))] = dict(n_clips=d["n_clips"], n_contaminated=d["n_contaminated"])
    ex = dict(
        n_vi_items=len(items), source_benchmarks=sorted(benches), dataset_excluded=DATASET_EXCLUDED, pools_used=POOLS_USED,
        ids={k: sorted(v) for k, v in ids.items()}, n_vatex_ytids=len(vatex),
        n_bench_charades={k: (len(v) if v and not k.startswith("_") else v) for k, v in bench_ch.items()},
        sha256_matches=sha_hits,
        phash=dict(n_clips_checked=ph["n_clips"], n_contaminated=ph["n_contaminated"], by_set=ph["by_set"],
                   contaminated=[dict(rel=c.get("rel") or c["path"], matches=c["matches"]) for c in ph["contaminated"]]),
    )
    json.dump(ex, open(f"{a.root}/decon/exclusions.json", "w"), indent=1)
    print(json.dumps({k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in ex.items()}, indent=1))
    print("v1.1 exact ids:", {k: len(v) for k, v in ids.items()}, "| other v1.1 sources:", len(other))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--root", default=os.path.expanduser("~/gv"))
    ap.add_argument("--name"); ap.add_argument("--raw")
    a = ap.parse_args()
    dict(sha=cmd_sha, build=cmd_build, bench_ids=cmd_bench_ids)[a.cmd](a)
