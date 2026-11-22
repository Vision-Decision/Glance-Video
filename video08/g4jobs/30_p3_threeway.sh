# same process, same stream state, same tokens: S1b head / zero-shot generative / SFT first token / SFT decoding;
# plus probe_val for temperature scaling
export GV=/content/gv PYTHONPATH=/content/gv/code TOKENIZERS_PARALLELISM=false
cd $GV; O=out/p3; pids=()
for f in 2 30; do python -m video08.evals.p3_eval --ckpt $O/s1b_lora16/ckpt_final.pt --sft $O/sft_lora16/ckpt_final.pt --fps $f --n 300 --out $O/eval3_s1b_sft_fps$f.jsonl > logs/p3x_$f.log 2>&1 & pids+=($!); done
python -m video08.evals.p3_eval --ckpt $O/s1b_lora16/ckpt_final.pt --sft $O/sft_lora16/ckpt_final.pt --fps 2 --n 150 --split probe_val --gen 0 --out $O/probeval3_fps2.jsonl > logs/p3x_pv.log 2>&1 & pids+=($!)
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; exit $rc
