"""Pack normalised clips into ~2 GB tar shards and push them, the records and the decontamination files to the
PRIVATE HF dataset Vision-Decision/glance-video-v0 (run on gen-host; HF token from the environment / cached login).

  python push_shards.py --root ~/gv --names k700val,charades,countix_train [--records] [--dry]
Shard members are stored as gv/clips/<name>/... so that extracting into data_root (= /content/data) gives the
paths used by the records. Already pushed shards (shards/manifest.json) are skipped; new clips go to new shards.
"""
import argparse, json, os, tarfile
from huggingface_hub import HfApi

REPO = "Vision-Decision/glance-video-v0"
SHARD_BYTES = 2 << 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/gv"))
    ap.add_argument("--names", default="")
    ap.add_argument("--records", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    api = HfApi()
    if not a.dry:
        api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
        assert api.repo_info(REPO, repo_type="dataset").private, "repo must be private"
    os.makedirs(f"{a.root}/shards", exist_ok=True)
    man_p = f"{a.root}/shards/manifest.json"
    man = json.load(open(man_p)) if os.path.exists(man_p) else {"shards": {}, "files": []}
    have = set(man["files"])
    base = os.path.dirname(a.root.rstrip("/"))
    for name in [n for n in a.names.split(",") if n]:
        idx = [json.loads(l)["path"] for l in open(f"{a.root}/clips/{name}/index.jsonl")]
        todo = [p for p in idx if os.path.relpath(p, base) not in have]
        k = sum(1 for s in man["shards"] if s.startswith(name + "-"))
        while todo:
            sname = f"{name}-{k:04d}.tar"
            tp = f"{a.root}/shards/{sname}"
            size, members = 0, []
            with tarfile.open(tp, "w") as tf:
                while todo and size < SHARD_BYTES:
                    p = todo.pop(0)
                    r = os.path.relpath(p, base)
                    tf.add(p, arcname=r)
                    size += os.path.getsize(p); members.append(r)
                tf.add(f"{a.root}/clips/{name}/index.jsonl", arcname=f"gv/clips/{name}/index.jsonl")
            print(sname, len(members), size >> 20, "MB", flush=True)
            if not a.dry:
                api.upload_file(path_or_fileobj=tp, path_in_repo=f"shards/{sname}", repo_id=REPO, repo_type="dataset",
                                commit_message=f"shard {sname}")
                os.remove(tp)
            man["shards"][sname] = len(members); man["files"] += members; have |= set(members)
            json.dump(man, open(man_p, "w"))
            k += 1
    if a.records and not a.dry:
        api.upload_folder(folder_path=f"{a.root}/records", path_in_repo="records", repo_id=REPO, repo_type="dataset",
                          commit_message="records")
        for f in ("exclusions.json", "report.md"):
            if os.path.exists(f"{a.root}/decon/{f}"):
                api.upload_file(path_or_fileobj=f"{a.root}/decon/{f}", path_in_repo=f"decon/{f}", repo_id=REPO,
                                repo_type="dataset")
        if os.path.exists(f"{a.root}/README.md"):
            api.upload_file(path_or_fileobj=f"{a.root}/README.md", path_in_repo="README.md", repo_id=REPO, repo_type="dataset")
        api.upload_file(path_or_fileobj=man_p, path_in_repo="shards/manifest.json", repo_id=REPO, repo_type="dataset")


if __name__ == "__main__":
    main()
