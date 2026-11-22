export GV=/content/gv PYTHONPATH=/content/gv/code TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $GV && rm -rf out/p3/sft_smoke && python -m video08.train_s1 --out out/p3/sft_smoke --gen_sft 1 --lora 16 --steps 6 --batch 4 --qbatch 8 --times 2 --write_s 2 --workers 4 --data_seed 0 --save 60 && test -f out/p3/sft_smoke/ckpt_final.pt
