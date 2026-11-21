"""Fetch every raw source for video08 onto a fresh host (hf-mirror first; big files with parallel range requests).
python video08/data/fetch_all.py DATA   (HF_ENDPOINT=https://hf-mirror.com; run with pylib on PYTHONPATH)"""
import os, subprocess, sys
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
from concurrent.futures import ThreadPoolExecutor

R = sys.argv[1]; RAW = f"{R}/raw"; EP = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
PDL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdl.py")
api = HfApi(endpoint=EP)


def big(repo, path, out, conns=16):
    if os.path.exists(out):
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    url = f"{EP}/datasets/{repo}/resolve/main/{path}"
    subprocess.run([sys.executable, PDL, url, out, str(conns)], check=True)


def repo_files(repo):
    return [(f.path, f.size) for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True) if hasattr(f, "size")]


def fetch(repo, out, big_min=200 << 20, conns=16, allow=None):
    files = repo_files(repo)
    small = [p for p, s in files if s < big_min and (allow is None or any(p.endswith(a) for a in allow))]
    if small:  # file by file: names like "x_[13].mp4" are glob patterns to allow_patterns
        import time as _t

        def get(p):
            for k in range(10):
                try:
                    return hf_hub_download(repo, p, repo_type="dataset", local_dir=out, endpoint=EP)
                except Exception as e:
                    _t.sleep(min(60, 3 * 2 ** k))
            print("gave up", repo, p, flush=True)
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(get, small))
    for p, s in files:
        if s >= big_min and (allow is None or any(p.endswith(a) for a in allow)):
            big(repo, p, f"{out}/{p}", conns); print("got", repo, p, flush=True)


what = sys.argv[2:] or ["timelens", "tl100k", "pushup", "repcount", "charades", "model"]
for w in what:
    if os.path.exists(f"{RAW}/{w}/.done"):
        continue
    if w == "timelens":
        fetch("TencentARC/TimeLens-Bench", f"{RAW}/timelens", allow=[".json", "README.md"])
    elif w == "tl100k":
        fetch("TencentARC/TimeLens-100K", f"{RAW}/tl100k", allow=[".jsonl", "README.md"])
    elif w == "pushup":
        fetch("anonymousatom/pushupbench", f"{RAW}/pushup")
    elif w == "repcount":
        fetch("lmms-lab-eval/repcounta-lance", f"{RAW}/repcount", conns=24)
    elif w == "charades":
        fetch("lmms-eval/charades_sta", f"{RAW}/charades_zip", conns=24)
        for z in sorted(p for p in os.listdir(f"{RAW}/charades_zip") if p.endswith(".zip")):
            subprocess.run(["unzip", "-q", "-o", f"{RAW}/charades_zip/{z}", "-d", f"{RAW}/charades"], check=True)
        ok = len(os.listdir(f"{RAW}/charades/Charades_v1_480"))
        print("charades videos", ok, flush=True)
        if ok >= 9848:
            subprocess.run(["rm", "-rf", f"{RAW}/charades_zip"])
    elif w == "hcstvg2":  # HC-STVG v2: videos + annos/{train,test}.json (14.6 GB copy; ModelScope's 115 GB zips not needed)
        fetch("ShijianW01/hc-stvg2", f"{RAW}/hcstvg2", conns=8)
    elif w == "mevis":
        fetch("FudanCVL/MeViS", f"{RAW}/mevis", conns=24)
        subprocess.run(["unzip", "-q", "-o", f"{RAW}/mevis/MeViS_release.zip", "-d", f"{RAW}/mevis"], check=True)
    elif w == "lasot":  # subset of classes with natural-language target descriptions ("where is X now")
        import shutil, zipfile
        for c in ("person", "car", "dog", "cup", "bicycle", "hand"):
            if os.path.exists(f"{RAW}/lasot/{c}/.done"):
                continue
            free = shutil.disk_usage(RAW).free / 2**30
            if free < 45:  # shared disk: keep >= 35 GB after this class (zip ~5 GB + 10 fps frames)
                print("disk guard: stop LaSOT before", c, round(free), "GB free", flush=True); break
            big("l-lt/LaSOT", f"{c}.zip", f"{RAW}/lasot/{c}.zip", 16)
            with zipfile.ZipFile(f"{RAW}/lasot/{c}.zip") as z:  # annotations + every 3rd frame (30 -> 10 fps)
                for n in z.namelist():
                    b = os.path.basename(n)
                    if b.endswith(".txt") or (b.endswith(".jpg") and int(os.path.splitext(b)[0]) % 3 == 1):
                        z.extract(n, f"{RAW}/lasot")
            os.remove(f"{RAW}/lasot/{c}.zip")
            os.makedirs(f"{RAW}/lasot/{c}", exist_ok=True); open(f"{RAW}/lasot/{c}/.done", "w").write("10fps")
        fetch("l-lt/LaSOT", f"{RAW}/lasot", allow=[".txt", "README.md"])
    elif w == "refcoco":  # annotations only; images = COCO train2014 already on the host
        fetch("jxu124/refcoco", f"{RAW}/refcoco")
    elif w == "model":
        snapshot_download("Qwen/Qwen3.5-0.8B", endpoint=EP)
    os.makedirs(f"{RAW}/{w}", exist_ok=True); open(f"{RAW}/{w}/.done", "w").write("ok")
    print("done", w, flush=True)
