"""Fetch the public raw data for the 0.8B video experiments (hf-mirror first; set HF_ENDPOINT).
python video08/data/download.py ROOT [set ...]   sets: charades timelens repcount pushup tl100k"""
import os, sys, subprocess, glob
from huggingface_hub import snapshot_download

ROOT = sys.argv[1]; SETS = sys.argv[2:] or ["timelens", "pushup", "repcount", "charades", "tl100k"]
R = f"{ROOT}/raw"
SPEC = {  # name -> (repo, allow_patterns)
    "timelens": ("TencentARC/TimeLens-Bench", ["*.json", "README.md"]),
    "tl100k": ("TencentARC/TimeLens-100K", ["*.jsonl", "README.md"]),
    "pushup": ("anonymousatom/pushupbench", None),
    "repcount": ("lmms-lab-eval/repcounta-lance", None),
    "repcount_ann": ("lmms-lab-eval/repcount", None),
    "charades": ("lmms-eval/charades_sta", None),
}
for s in SETS + (["repcount_ann"] if "repcount" in SETS else []):
    repo, pat = SPEC[s]; out = f"{R}/{s}"
    if os.path.exists(f"{out}/.done"):
        print("skip", s); continue
    for k in range(5):
        try:
            snapshot_download(repo, repo_type="dataset", allow_patterns=pat, local_dir=out, max_workers=8); break
        except Exception as e:
            print("retry", s, k, repr(e)[:200], flush=True)
    else:
        raise SystemExit(f"failed {s}")
    if s == "charades":  # unzip the four parts into raw/charades/Charades_v1_480, then drop the zips
        for z in sorted(glob.glob(f"{out}/Charades_v1_480_part_*.zip")):
            subprocess.run(["python", "-c", f"import zipfile;zipfile.ZipFile('{z}').extractall('{out}')"], check=True)
            os.remove(z)
    open(f"{out}/.done", "w").write("ok")
    print("done", s, flush=True)
