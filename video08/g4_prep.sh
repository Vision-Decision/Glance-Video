#!/usr/bin/env bash
# G4 day prep (CPU/network only, runs next to M7/M8): Charades train videos, RepCount-A (all splits) for S1 resume.
export GV=/content/gv PYTHONPATH=/content/gv/code HF_HUB_DISABLE_PROGRESS_BARS=1
cd $GV; step() { echo "$(date +%H:%M:%S) $*" >> logs/progress.log; }
pip -q install pylance > logs/pip_lance.log 2>&1
if [ ! -f data/raw/charades/.train_done ]; then
  python code/video08/data/pdl.py https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip data/raw/charades.zip 24 > logs/pdl2.log 2>&1
  (cd data/raw/charades && xargs -a $GV/data/train_ids.txt unzip -q -o ../charades.zip) && touch data/raw/charades/.train_done && rm -f data/raw/charades.zip
fi
step "charades videos now: $(ls data/raw/charades/Charades_v1_480 | wc -l)"
python -c "from huggingface_hub import snapshot_download; snapshot_download('lmms-lab-eval/repcounta-lance', repo_type='dataset', local_dir='data/raw/repcount', max_workers=8)" > logs/rep.log 2>&1
python code/video08/data/repcount.py $GV/data test validation train >> logs/rep.log 2>&1; step "repcount: $(wc -l < data/idx/repcount_train.jsonl) train"
step G4-PREP-DONE
