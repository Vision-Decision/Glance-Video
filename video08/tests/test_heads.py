"""CPU smoke of the S1 readout: stream a few groups, build every task's questions, forward heads (LoRA on), losses,
backward; check that only readout/LoRA params get gradients and that the stream state is unchanged by a readout.
python video08/tests/test_heads.py VIDEO.mp4"""
import sys, os, random
sys.modules["fla"] = None
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from video08.stream import Stream, load
from video08 import vio, heads
from video08.tasks import Q
import json

model, tok = load(device="cpu", dtype=torch.float32)
lora = heads.add_lora(model.model.language_model, r=4)
ro = heads.Readout(model.model.language_model.embed_tokens.weight, layers=(12, 24), slots=4, bins=32)
D = os.path.expanduser("~/glance-video-exp/data")
classes = json.load(open(f"{D}/idx/classes.json"))
mapping = {l.split()[0]: l.split()[1:] for l in open(f"{D}/meta/ann/Charades/Charades_v1_mapping.txt")}
qb = Q(tok, classes, mapping)
row = dict(actions=[["c092", 0.5, 2.0], ["c147", 0.0, 1.0], ["c092", 2.5, 3.0]], sta=[["person cooks food.", 0.5, 2.0]],
           reps=[[0, 1.0], [1.0, 2.0]], action="push_up")
fr, tt = vio.decode(sys.argv[1], 2.0, 256, 0, 3.0)
st = Stream(model, tok)
vis, grid = st.encode(vio.pairs(fr))
with torch.no_grad():
    st.write(vis, [float(x) for x in tt], grid)
snap = [L.recurrent_states[0].clone() for L in st.cache.layers if hasattr(L, "recurrent_states") and L.recurrent_states[0] is not None]
kv0 = st.kv_len()
rng = random.Random(0); t = float(tt[-1])
items = qb.presence(row, t, rng) + qb.temporal(row, t, rng) + qb.choice(row, t, rng) + qb.count(row, t, rng)
print("items", len(items), {k: sum(1 for x in items if x["task"] == k) for k in ("presence", "temporal", "choice", "count")})
heads.LORA["on"] = True
res = ro(st, [x["ids"] for x in items], [x.get("ends") for x in items])
heads.LORA["on"] = False
sys.argv = [sys.argv[0], "--out", "/tmp/x"]  # train_s1's losses() needs its module globals; reimplement the minimal part
import torch.nn.functional as F
pres = [i for i, x in enumerate(items) if x["task"] == "presence"]
L = F.binary_cross_entropy_with_logits(res["presence"][pres], torch.tensor([items[i]["y"] for i in pres]).float())
tem = [i for i, x in enumerate(items) if x["task"] == "temporal"]
L = L + torch.stack([heads.slot_loss(res["obj"][i], res["start"][i], res["end"][i], items[i]["segs"], 32) for i in tem]).mean()
ch = [i for i, x in enumerate(items) if x["task"] == "choice"]
L = L + F.cross_entropy(res["choice"][ch[0]][None], torch.tensor([items[ch[0]]["y"]]))
cn = [i for i, x in enumerate(items) if x["task"] == "count"]
L = L + F.cross_entropy(res["count"][cn], torch.tensor([items[i]["y"] for i in cn]))
L.backward()
g_ro = sum(p.grad is not None and p.grad.abs().sum() > 0 for p in ro.parameters())
g_lora = sum(p.grad is not None and p.grad.abs().sum() > 0 for p in lora)
g_base = sum(p.grad is not None for n, p in model.named_parameters() if not (n.endswith(".A") or n.endswith(".B")))
print("loss", float(L), "ro params with grad", g_ro, "/", len(list(ro.parameters())), "lora with grad", g_lora, "/", len(lora), "base params with grad", g_base)
same = all(torch.equal(a, L_.recurrent_states[0]) for a, L_ in zip(snap, [L_ for L_ in st.cache.layers if hasattr(L_, "recurrent_states") and L_.recurrent_states[0] is not None]))
print("stream state unchanged by readout:", same, "kv", kv0, st.kv_len())
print("decode", heads.decode_slots(res["obj"][tem[0]], res["start"][tem[0]], res["end"][tem[0]], 32)[:2])
