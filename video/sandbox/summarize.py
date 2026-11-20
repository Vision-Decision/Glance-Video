#!/usr/bin/env python3
"""Summarize a sandbox run: accuracy, no-answers, time, tokens, cost-equivalent, comparison with the published table.

  python3 video/sandbox/summarize.py --run full1 [--model deepseek-v4.1-flash]
Tokens come from the gateway log (data/video_index/runs/_gateway/usage.jsonl), summed per call.
Cost: DeepSeek = cost *equivalent* at DeepSeek's list price (Ollama cloud is a flat subscription);
Gemini = Gemini API list price (billed); Claude = Claude Code's reported cost (API-equivalent, subscription here).
Lower bound uses the cached price for cache_read tokens; upper bound charges them at the full input price,
like the official table's Gemini/OpenAI bounds.
"""
import argparse, collections, csv, json, os, statistics as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESEARCH = os.environ.get("VI_RESEARCH", os.path.join(ROOT, "data/research/video_index_v1_1"))
RUNS = os.path.join(os.environ.get("VI_DATA", os.path.join(ROOT, "data/video_index")), "runs")
ITEMS = os.path.join(RESEARCH, "data/test.jsonl")
SCORES = next((p for p in (os.path.join(RESEARCH, "scores.csv"),
                           os.path.join(ROOT, "data/research/Video-Index-repo/v1.1/scores.csv")) if os.path.exists(p)), "")
# $/token: (uncached input, cached input, output). Third-party list prices, Nov 2026 -- verify before publishing.
PRICES = {"deepseek-v4.1-flash": (0.30e-6, 0.006e-6, 1.20e-6),   # DeepSeek API peak price
          "gemini-4-hydrogen": (1.50e-6, 0.075e-6, 7.50e-6)}     # Gemini API standard tier; thinking billed as output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    a = ap.parse_args()
    I = {json.loads(l)["item_id"]: json.loads(l) for l in open(ITEMS)}
    d = os.path.join(RUNS, a.run, a.model.replace(":", "_"))
    R = {i: json.load(open(os.path.join(d, i, "result.json"))) for i in sorted(os.listdir(d))
         if os.path.exists(os.path.join(d, i, "result.json"))}
    tok = collections.defaultdict(lambda: collections.Counter())
    nreq = collections.Counter()
    rewritten = 0
    glog = os.path.join(RUNS, "_gateway", "usage.jsonl")
    if os.path.exists(glog):
        for l in open(glog):
            r = json.loads(l)
            run, _, iid = r["call"].partition(".")
            if iid not in R or run != str(R[iid].get("run")):
                continue  # usage is keyed by the run that actually executed the item (reused items keep theirs)
            nreq[iid] += 1
            rewritten += r.get("requested_model") != r.get("model")
            u = r.get("usage") or {}
            if "promptTokenCount" in u:  # Gemini API usageMetadata
                cached = u.get("cachedContentTokenCount", 0) or 0
                tok[iid]["input_tokens"] += (u.get("promptTokenCount", 0) or 0) - cached
                tok[iid]["cache_read_input_tokens"] += cached
                tok[iid]["output_tokens"] += (u.get("candidatesTokenCount", 0) or 0) + (u.get("thoughtsTokenCount", 0) or 0)
            else:
                for k in ("input_tokens", "output_tokens", "cache_read_input_tokens"):
                    tok[iid][k] += u.get(k, 0) or 0
    rows = []
    for i, r in R.items():
        t = tok[i]
        if a.model in PRICES:
            P_IN, P_CACHE, P_OUT = PRICES[a.model]
            lo = t["input_tokens"] * P_IN + t["cache_read_input_tokens"] * P_CACHE + t["output_tokens"] * P_OUT
            hi = (t["input_tokens"] + t["cache_read_input_tokens"]) * P_IN + t["output_tokens"] * P_OUT
        else:  # Claude: the cost Claude Code reports (API-equivalent; billed to the subscription here)
            lo = hi = r.get("cc_reported_cost_usd") or 0
            u = r.get("usage") or {}
            t = collections.Counter({"input_tokens": u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0),
                                     "cache_read_input_tokens": u.get("cache_read_input_tokens", 0),
                                     "output_tokens": u.get("output_tokens", 0)})
        rows.append(dict(item_id=i, group=I[i]["capability_group"], benchmark=I[i]["benchmark"],
                         dur_video=I[i]["duration_s"], status=r["status"], pred=r["prediction"], key=r["expected"],
                         correct=int(r["correct"]), wall_s=r["duration_s"], n_tool_calls=r["n_tool_calls"],
                         requests=nreq[i], in_tok=t["input_tokens"], cached_tok=t["cache_read_input_tokens"],
                         out_tok=t["output_tokens"], cost_lo=round(lo, 4), cost_hi=round(hi, 4)))
    n = len(rows)
    if not n:
        return print("no results")
    with open(os.path.join(RUNS, a.run, "per_question.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    acc = 100 * sum(r["correct"] for r in rows) / n
    noans = collections.Counter(r["status"] for r in rows if r["status"] != "succeeded")
    wall = [min(r["wall_s"], 3600) for r in rows]
    s = {"run": a.run, "model": a.model, "n": n, "correct": sum(r["correct"] for r in rows), "acc": round(acc, 1),
         "no_answer": dict(noans), "time_min_mean": round(st.mean(wall) / 60, 1),
         "time_min_median": round(st.median(wall) / 60, 1),
         "tokens_mean": {k: round(st.mean(r[k] for r in rows)) for k in ("in_tok", "cached_tok", "out_tok")},
         "cost_per_q_usd_equiv": [round(st.mean(r["cost_lo"] for r in rows), 3), round(st.mean(r["cost_hi"] for r in rows), 3)],
         "requests_mean": round(st.mean(r["requests"] for r in rows), 1), "model_rewrites": rewritten,
         "by_group": {g: f"{sum(r['correct'] for r in rows if r['group']==g)}/{sum(r['group']==g for r in rows)}"
                      for g in sorted({r['group'] for r in rows})},
         "pred_letters": dict(collections.Counter(r["pred"] or "-" for r in rows)),
         "key_is_last_and_pred_last": sum(1 for r in rows if r["pred"] and r["pred"] == chr(64 + len(I[r["item_id"]]["options"]))),
         }
    if n == 144 and SCORES:
        sc = list(csv.DictReader(open(SCORES)))
        k = [c for c in sc[0] if "144" in c or "ccuracy" in c][0]
        s["published_rank"] = 1 + sum(float(x[k]) > acc for x in sc)
    json.dump(s, open(os.path.join(RUNS, a.run, "summary.json"), "w"), indent=1)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
