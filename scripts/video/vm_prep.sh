#!/usr/bin/env bash
# Glance-Video data prep on a Colab VM (48 CPUs): Kinetics-700 val + Charades normalisation, motion stats, pHash decon,
# records, shards -> PRIVATE HF dataset Vision-Decision/glance-video-v0. gen-host supplies meta/ (exclusions, annotations,
# VI frame hashes) and the countix_train / repcount shards through the same HF dataset.
# Usage (VM): bash /content/dv/scripts/video/vm_prep.sh > /content/gv_prep.log 2>&1
set -uo pipefail
export HF_HUB_DISABLE_PROGRESS_BARS=1
R=/content/gv; S=/content/dv/scripts/video; W=${W:-40}
mkdir -p $R/raw/k700val $R/raw/charades $R/decon $R/logs
cp $S/vhash.py $S/prep_clips.py $S/clip_stats.py $S/gen_records.py $S/push_shards.py $S/decon_list.py $R/
cd $R
echo "[prep] $(date -u +%T) meta"
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("Vision-Decision/glance-video-v0", repo_type="dataset", local_dir="/content/hf_gv", allow_patterns=["meta/*"])
EOF
cp -r /content/hf_gv/meta/ann $R/raw/ 2>/dev/null; cp -r /content/hf_gv/meta/decon/* $R/decon/ 2>/dev/null; ls $R/raw/ann $R/decon
echo "[prep] $(date -u +%T) download"
which aria2c >/dev/null || apt-get install -y -qq aria2 >/dev/null 2>&1
( cd $R/raw/k700val && wget -q -c https://s3.amazonaws.com/kinetics/700_2020/val/k700_2020_val_path.txt && \
  (cat k700_2020_val_path.txt | xargs -P 16 -n 1 wget -q -c; true) && \
  echo "[prep] $(date -u +%T) k700 tars: $(ls $R/raw/k700val/*.tar.gz | wc -l)" && \
  python $R/prep_clips.py k700 --root $R --workers $((W*3/4)) > $R/logs/k700.log 2>&1 && \
  python $R/clip_stats.py --root $R --name k700val --workers $((W*3/4)) > $R/logs/stats.log 2>&1 ) &
( cd $R/raw/charades && aria2c -q -x 16 -s 16 -c https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip && \
  echo "[prep] $(date -u +%T) charades zip $(du -sh Charades_v1_480.zip)" && cp $R/raw/ann/Charades.zip $R/raw/charades/ && \
  cd $R && python prep_clips.py charades --root $R --workers $((W/4)) > $R/logs/charades.log 2>&1 ) &
wait
echo "[prep] $(date -u +%T) clips: $(wc -l clips/*/index.jsonl | tail -1)"
cd $R
until [ -f $R/decon/vi_hashes.npz ]; do   # VI fingerprints come from gen-host via meta/decon on HF
  python -c "from huggingface_hub import hf_hub_download as h; h('Vision-Decision/glance-video-v0','meta/decon/vi_hashes.npz',repo_type='dataset',local_dir='/content/hf_gv')" >/dev/null 2>&1 \
    && cp /content/hf_gv/meta/decon/vi_hashes.npz $R/decon/ || { echo "[prep] waiting for vi_hashes"; sleep 120; }
done
python prep_clips.py decon --root $R --name k700val,charades --workers $W > logs/decon.log 2>&1
python decon_list.py build --root $R > logs/decon_build.log 2>&1; tail -3 logs/decon_build.log
echo "[prep] $(date -u +%T) shards"
python push_shards.py --root $R --names k700val,charades > logs/push.log 2>&1; tail -2 logs/push.log
echo "[prep] $(date -u +%T) DONE"
