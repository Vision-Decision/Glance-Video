# generative-SFT control: same data distribution, question-block LoRA r=16, steps, batch, write shape as S1b; answers as text
export GV=/content/gv PYTHONPATH=/content/gv/code TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $GV && python -m video08.train_s1 --out out/p3/sft_lora16 --gen_sft 1 --lora 16 --steps 1200 --batch 4 --qbatch 8 --times 2 --write_s 2 --workers 8 --data_seed 0 && test -f out/p3/sft_lora16/ckpt_final.pt
