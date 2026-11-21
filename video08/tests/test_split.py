"""GPU: split attention (no boolean mask) == SDPA with mask, on a streamed clip with bounded KV + chunked writes + asks,
and speed of a 15-group write / 16-question ask under both.  python -m video08.tests.test_split VIDEO"""
import sys, time, torch
from video08.stream import Stream, load, yes_no_ids, p_yes
from video08 import vio, feed
from video08.attn import register
register()
model, tok = load(attn="sdpa"); yid, nid = yes_no_ids(tok)
fr, tt = vio.decode(sys.argv[1], 30, 256, 0, 20.0)
qs = [f"Is someone doing thing number {i} right now? Answer yes or no." for i in range(16)]
out = {}
for impl in ("sdpa", "split"):
    model.set_attn_implementation(impl)
    st = Stream(model, tok, full_s=4, sub_s=10)
    torch.cuda.synchronize(); t = time.perf_counter()
    _, tm, _ = feed.feed(st, fr, tt, chunk=15)
    torch.cuda.synchronize(); w = time.perf_counter() - t
    lg = st.ask(qs)["logits"]; torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(10):
        st.ask(qs)
    torch.cuda.synchronize(); a = (time.perf_counter() - t) / 10
    out[impl] = (lg, p_yes(lg, yid, nid), st.generate("What is the person doing? Answer briefly."))
    print(impl, "write total s", round(w, 2), "per 15-group write ms", round(tm["lm"] / len(tm["step_ms"]) * 1e3, 1), "ask16 ms", round(a * 1e3, 1), "kv", st.kv_len(), out[impl][2])
print("max|dlogit|", (out["sdpa"][0] - out["split"][0]).abs().max().item(), "max|dp|", (out["sdpa"][1] - out["split"][1]).abs().max().item())
