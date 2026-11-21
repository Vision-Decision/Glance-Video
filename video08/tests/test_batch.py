"""GPU: a lockstep batched stream (B=3, different lengths, rows dropped as they end) gives the same answers as three
separate streams.  python -m video08.tests.test_batch V1 V2 V3"""
import sys, torch
from video08.stream import Stream, load, yes_no_ids, p_yes
from video08 import vio, feed
model, tok = load(); yid, nid = yes_no_ids(tok)
q = ["Is there a person in the video? Answer yes or no.", "Is there a dog in the video? Answer yes or no."]
vids = [vio.decode(p, 30, 256, 0, L) for p, L in zip(sys.argv[1:4], (6.0, 9.0, 7.5))]
st = Stream(model, tok, full_s=2, sub_s=4)
single = []
for fr, tt in vids:
    st.reset(); feed.feed(st, fr, tt, chunk=15); single.append(p_yes(st.ask(q)["logits"], yid, nid))
got = {}
def on_end(i, row):
    got[i] = p_yes(st.ask(q, rows=[row, row])["logits"], yid, nid)
feed.feed_batch(st, vids, 15, on_end)
for i in range(3):
    print(i, single[i].tolist(), got[i].tolist(), "max|dp|", float((single[i] - got[i]).abs().max()))
