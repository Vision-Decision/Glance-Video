#!/usr/bin/env python3
"""Results snapshot for the Video-Index replica runs (run on gen-host; prints Markdown).

  VI_DATA=... VI_RESEARCH=... python3 sandbox/snapshot.py > SNAPSHOT.md
Conditions pool several runs of the same config. Scoring = our README rule: correct iff status == succeeded and
letter == key (a timeout is wrong even with a letter). Rate-limited (Opus) and compromised (vdm outage) items have
no result.json in the run dirs and are therefore not scored. Paired comparisons use only items finished in both
conditions; 95% CI = paired bootstrap over items (10,000 resamples, seed 0).
"""
import collections, datetime as dt, glob, json, os, random, re, statistics as st

H = os.environ.get("VI_HOME", "~/videoindex")
RUNS = os.path.join(os.environ.get("VI_DATA", f"{H}/data"), "runs")
RESEARCH = os.environ.get("VI_RESEARCH", f"{H}/research")
ITEMS = {json.loads(l)["item_id"]: json.loads(l) for l in open(f"{RESEARCH}/data/test.jsonl")}
HARD = set(json.load(open(f"{H}/sandbox/hard60.json")))
OFFICIAL = json.load(open(f"{RESEARCH}/official_items.json")) if os.path.exists(f"{RESEARCH}/official_items.json") else {}
GPRICE = (1.50e-6, 0.075e-6, 7.50e-6)      # Gemini API list price (in, cached in, out incl. thinking) $/token
DPRICE = (0.30e-6, 0.006e-6, 1.20e-6)      # DeepSeek list price (cost-equivalent; Ollama cloud is flat-rate)

COND = [  # name, model dir, runs
    ("DeepSeek V4.1 Flash, no tool", "deepseek-v4.1-flash", ["pilot10", "full1"]),
    ("DeepSeek V4.1 Flash, +vdm", "deepseek-v4.1-flash", ["ds_vdm_hard"]),
    ("Gemini 4 Hydrogen, no tool", "gemini-4-hydrogen", ["gem_pilot5", "gem_hard25"]),
    ("Gemini 4 Hydrogen, +vdm", "gemini-4-hydrogen", ["gem_vdm_pilot5", "gem_vdm_hard25"]),
    ("Claude Opus 5.5 (subscription), no tool", "claude-opus-5-5", ["opus_pilot5", "opus_hard10"]),
    ("Claude Opus 5.5 (subscription), +vdm", "claude-opus-5-5", ["opus_vdm_hard10"]),
]
PAIRS = [(0, 1), (2, 3), (4, 5), (0, 2), (0, 4), (2, 4)]
OFFICIAL_REF = [(4, "claude-opus-5-5"), (2, "gemini-4-argon"), (2, "gemini-3.8-flash"), (0, "claude-haiku-4-5"),
                (0, "claude-haiku-5-5")]


def load(model, runs):
    out, running, rl = {}, 0, 0
    for run in runs:
        for d in glob.glob(os.path.join(RUNS, run, model, "*")):
            f = os.path.join(d, "result.json")
            if os.path.exists(f):
                r = json.load(open(f))
                out[r["item_id"]] = r
            elif os.path.exists(os.path.join(d, "claimed")):
                running += 1
            if os.path.exists(os.path.join(d, "rate_limited.json")) and not os.path.exists(f):
                rl += 1
    return out, running, rl


def boot(d, n=10000):
    rng = random.Random(0)
    m = len(d)
    b = sorted(st.mean(d[rng.randrange(m)] for _ in range(m)) for _ in range(n))
    return b[int(.025 * n)], b[int(.975 * n) - 1]


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "–"


def fmt_min(xs):
    return f"{st.mean(xs) / 60:.1f} / {st.median(xs) / 60:.1f}" if xs else "–"


def vdm_cmds(d):
    """Number of vdm sub-commands the agent issued (from tool calls in stdout.jsonl)."""
    n = 0
    pat = re.compile(r"(?:^|[\s;&|(`])vdm\s+(find|count|verify|locate|motion|choose)\b")
    f = os.path.join(d, "stdout.jsonl")
    if not os.path.exists(f):
        return 0
    for line in open(f, errors="ignore"):
        if '"tool_use"' not in line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        calls = []
        if ev.get("type") == "assistant":
            calls = [c.get("input") for c in (ev.get("message") or {}).get("content", []) if c.get("type") == "tool_use"]
        elif ev.get("type") == "tool_use":
            calls = [ev.get("parameters") or ev.get("input")]
        for c in calls:
            n += len(pat.findall(json.dumps(c or {})))
    return n


def main():
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    P = []
    p = P.append
    p(f"# Video-Index replica: results snapshot ({now})\n")
    p("Partial results. Every run walks the same seeded random order (random.Random(0) over sorted item ids, on "
      "the 144 v1.1 items or on the 64-item hard set), so a partial run is a random subset of its set. Scoring: "
      "correct = answered with the key; a timeout or a missing answer is wrong (README rule). Opus items stopped by "
      "the subscription rate limit and +vdm items hit by the vdm server outage are **not scored** and are re-queued. "
      "Hard set = `sandbox/hard60.json` (64 items, perception-bound / timeout-heavy for the official agents), so "
      "accuracies on it are far below the full-set numbers.\n")

    # ---------------------------------------------------------------- per condition
    R = []
    p("## 1. Accuracy per condition\n")
    p("| Condition | Runs | Scored n | Correct | Acc. | of which hard set | Running | Timeouts | Errors | Min/item mean / median |")
    p("|---|---|---|---|---|---|---|---|---|---|")
    for name, model, runs in COND:
        res, running, rl = load(model, runs)
        R.append(res)
        rs = list(res.values())
        c = sum(r["correct"] for r in rs)
        hs = [r for r in rs if r["item_id"] in HARD]
        to = sum(r["status"] == "timeout" for r in rs)
        er = sum(r["status"] == "error" for r in rs)
        p(f"| {name} | {', '.join(runs)} | {len(rs)} | {c} | {pct(c, len(rs))} | {sum(r['correct'] for r in hs)}/{len(hs)} "
          f"| {running} | {to} | {er} | {fmt_min([min(r['duration_s'], 3600) for r in rs])} |")
    p("")
    ds_full = {i: r for i, r in R[0].items()}
    p(f"DeepSeek no-tool on the full v1.1 set so far: {sum(r['correct'] for r in ds_full.values())}/{len(ds_full)} "
      f"({pct(sum(r['correct'] for r in ds_full.values()), len(ds_full))}), of 144. For reference, the published "
      "table: Opus 5.5 47.9%, Gemini 4 Argon 41.0%, Haiku 5.5 27.8%, Haiku 4.5 14.6%; chance ≈ 24.8%.\n")
    by = collections.defaultdict(lambda: [0, 0])
    for r in ds_full.values():
        by[ITEMS[r["item_id"]]["capability_group"]][0] += r["correct"]
        by[ITEMS[r["item_id"]]["capability_group"]][1] += 1
    p("By group (DeepSeek no-tool): " + ", ".join(f"{g} {a}/{b}" for g, (a, b) in sorted(by.items())) + "\n")

    # ---------------------------------------------------------------- paired
    p("## 2. Paired comparisons (common finished items only)\n")
    p("| B vs A | n common | Acc. A | Acc. B | Δ = B − A (pp) | 95% bootstrap CI | B fixes / B breaks |")
    p("|---|---|---|---|---|---|---|")

    def pair_row(la, A, lb, B):
        ids = sorted(set(A) & set(B))
        if not ids:
            return p(f"| {lb} vs {la} | 0 | – | – | – | – | – |")
        a = [int(A[i]) for i in ids]
        b = [int(B[i]) for i in ids]
        d = [y - x for x, y in zip(a, b)]
        lo, hi = boot(d)
        p(f"| {lb} vs {la} | {len(ids)} | {sum(a)}/{len(ids)} | {sum(b)}/{len(ids)} | {100 * st.mean(d):+.1f} | "
          f"[{100 * lo:+.1f}, {100 * hi:+.1f}] | {sum(y > x for x, y in zip(a, b))} / {sum(x > y for x, y in zip(a, b))} |")
    for i, j in PAIRS:
        pair_row(COND[i][0], {k: r["correct"] for k, r in R[i].items()}, COND[j][0], {k: r["correct"] for k, r in R[j].items()})
    for i, om in OFFICIAL_REF:
        if om in OFFICIAL:
            pair_row(f"official {om} (published trace)", {k: v["c"] for k, v in OFFICIAL[om].items()},
                     COND[i][0], {k: r["correct"] for k, r in R[i].items()})
    p("\nWith n this small, a CI that spans 0 means no detectable effect; one fix or break moves Δ by 100/n pp. "
      "Official rows are a replication check: same items, same prompts, published traces (official rule credits a "
      "letter given at timeout; ours does not, which matters for at most a few items).\n")

    # ---------------------------------------------------------------- cost
    p("## 3. Cost and usage\n")
    gl = os.path.join(RUNS, "_gateway", "usage.jsonl")
    gem, ds, vdm = collections.defaultdict(float), collections.defaultdict(float), []
    for l in open(gl):
        try:
            r = json.loads(l)
        except Exception:
            continue
        call = r.get("call", "")
        if call == "vdm":
            vdm.append(r)
            continue
        if call == "connect":
            continue
        run = call.split(".")[0]
        u = r.get("usage") or {}
        if "promptTokenCount" in u or run.startswith("gem"):
            cached = u.get("cachedContentTokenCount", 0) or 0
            gem[run] += ((u.get("promptTokenCount", 0) or 0) - cached) * GPRICE[0] + cached * GPRICE[1] + \
                ((u.get("candidatesTokenCount", 0) or 0) + (u.get("thoughtsTokenCount", 0) or 0)) * GPRICE[2]
        else:
            ds[run] += (u.get("input_tokens", 0) or 0) * DPRICE[0] + (u.get("cache_read_input_tokens", 0) or 0) * DPRICE[1] \
                + (u.get("output_tokens", 0) or 0) * DPRICE[2]
    p(f"- **Gemini API (billed, list price, from gateway token counts): ${sum(gem.values()):.2f} total** — "
      + ", ".join(f"{k} ${v:.2f}" for k, v in sorted(gem.items())) + ". Budget ≈ $100, of which ≈ $65 reserved for vdm v1.")
    p(f"- DeepSeek (Ollama cloud flat subscription; cost-equivalent at DeepSeek list price): ${sum(ds.values()):.2f} — "
      + ", ".join(f"{k} ${v:.2f}" for k, v in sorted(ds.items()) if v > 0.005) + ".")
    # opus: CC-reported cost + latest rate-limit utilization
    cc, last = 0.0, None
    for f in glob.glob(os.path.join(RUNS, "opus_*", "*", "*", "stdout.jsonl")):
        rj = os.path.join(os.path.dirname(f), "result.json")
        if os.path.exists(rj):
            cc += json.load(open(rj)).get("cc_reported_cost_usd") or 0
        for line in open(f, errors="ignore"):
            if "rate_limit_event" in line:
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                info = ev.get("rate_limit_info") or {}
                w = info.get("unifiedWindows") or {}
                key = os.path.getmtime(f)
                if w and (last is None or key >= last[0]):
                    last = (key, w)
    if last:
        w = last[1]
        p(f"- Claude Opus 5.5 (subscription, not billed): Claude-Code-reported API-equivalent ${cc:.2f} for scored items; "
          f"latest rate-limit event: 5-hour window {100 * w.get('five_hour', {}).get('utilization', 0):.0f}%, "
          f"7-day window {100 * w.get('seven_day', {}).get('utilization', 0):.0f}% (cap for this study: ~60%).")
    p("")

    # ---------------------------------------------------------------- vdm calls
    p("## 4. vdm usage\n")
    if vdm:
        stc = collections.Counter(r["status"] for r in vdm)
        ok = [r["dt"] for r in vdm if r["status"] == 200 and r.get("path", "").startswith("/vdm/score")]
        p(f"- Gateway-level vdm HTTP requests (all +vdm runs and health checks): {len(vdm)}; status {dict(stc)}. "
          f"`/score` latency mean {st.mean(ok):.1f} s, median {st.median(ok):.1f} s, p90 "
          f"{sorted(ok)[int(.9 * len(ok)) - 1] if ok else 0:.1f} s (n={len(ok)}).")
        fails = [dt.datetime.fromtimestamp(r["ts"], dt.timezone.utc).strftime("%H:%M") for r in vdm if r["status"] != 200]
        if fails:
            p(f"- Failed vdm requests (599 = no server answered) at UTC {', '.join(sorted(set(fails)))}.")
    for idx in (1, 3, 5):
        name, model, runs = COND[idx]
        ns = []
        for run in runs:
            for d in glob.glob(os.path.join(RUNS, run, model, "*")):
                if os.path.exists(os.path.join(d, "result.json")):
                    ns.append(vdm_cmds(d))
        if ns:
            p(f"- {name}: vdm sub-commands per scored item mean {st.mean(ns):.1f}, median {st.median(ns)}, "
              f"items with no vdm call {sum(n == 0 for n in ns)}/{len(ns)}.")
    comp = sorted(os.path.basename(x) for x in glob.glob(os.path.join(RUNS, "_compromised", "*")) if os.path.isdir(x))
    p(f"- Compromised by the vdm outage and re-queued (not scored): {len(comp)} — " + ", ".join(comp) + ".")
    p("")

    # ---------------------------------------------------------------- choose-only
    p("## 5. vdm v0 choose-only (no LLM): v1.1 vs v1.0\n")
    p("`vdm choose`, 32 uniform frames, each frame answers the MCQ alone, mean option probability; **argmin** = pick "
      "the option with the lowest mean probability. Chance = mean of 1/#options.\n")
    p("**v1.0 is not a clean control.** Video-Index v1.0 was itself built adversarially against frame-based VLMs "
      "(repo `docs/pool.md`, `docs/compose.md`): the pool screen removes every item that an 8B attacker answers "
      "correctly with margin > log 2 from a **single frame** (frame 16 of 32), or a 2B attacker from 32 frames or "
      "32 shuffled frames, or from text only. The 840 released items are then the *hardest first*, ranked by the worst-case "
      "attacker margin percentile (median 0.14, max 0.23). Margin = log p(key) − mean log p(other options), so the "
      "released items are those where the frame models put *low* probability on the key. A single-frame scorer like "
      "Glance should therefore score below chance on v1.0 too. v1.1 (a subset that agents fail) inherits the effect. "
      "A clean control needs unscreened items from the source benchmarks.\n")
    p("| Set | Group | n | choose (argmax) | argmin | chance |")
    p("|---|---|---|---|---|---|")
    for label, f, total in (("v1.1", f"{H}/vdm/results/choose_only.jsonl", 144),
                            ("v1.0", f"{H}/vdm/results/choose_only_v1_0.jsonl", 840)):
        if not os.path.exists(f):
            continue
        recs = [r for r in map(json.loads, open(f)) if r.get("pred") and r.get("mean_prob")]
        groups = collections.defaultdict(list)
        for r in recs:
            groups[r["group"]].append(r)
            groups["all"].append(r)
        for g in sorted(groups, key=lambda g: (g != "all", g)):
            rs = groups[g]
            am = sum(min(r["mean_prob"], key=r["mean_prob"].get) == r["key"] for r in rs)
            ch = st.mean(1 / r["n_options"] for r in rs)
            p(f"| {label}{' (partial ' + str(len(recs)) + '/' + str(total) + ')' if g == 'all' and len(recs) < total else ''} "
              f"| {g} | {len(rs)} | {sum(r['correct'] for r in rs)} ({pct(sum(r['correct'] for r in rs), len(rs))}) "
              f"| {am} ({pct(am, len(rs))}) | {100 * ch:.1f}% |")
    p("")
    syn_f, perm_f = f"{H}/vdm/results/audit/synthetic.json", f"{H}/vdm/results/audit/permutation.json"
    if os.path.exists(syn_f):
        syn = json.load(open(syn_f))
        p(f"**Choose-only bug audit (independent, 07:50–08:10).** Synthetic clip (big colored rectangle, key at every "
          f"position, 2–7 options, same docker + v0 client path): argmax correct "
          f"{sum(s['pred'] == s['key'] for s in syn)}/{len(syn)}, mean p(key) "
          f"{st.mean(s['p_key'] for s in syn):.3f}. Server↔Glance.ask: same prompt builder (`typed_block`), same label "
          "groups 0..n−1 = A.., same softmax/temperature; server selftest compares the batched path with Glance.ask. "
          "Keys: test.jsonl `answer` = LETTERS[answer_idx] for 144/144 and = official `expected` for 144/144; the "
          "choose-only harness keeps the item's option order (no shuffling).")
    if os.path.exists(perm_f):
        perm = json.load(open(perm_f))
        same = sum(x["fwd_argmax"] == x["rev_argmax_original_letter"] for x in perm)
        lowkey = sum(x["rev_mapped_to_original"][x["key"]] <= sorted(x["rev_mapped_to_original"].values())[
            len(x["rev_mapped_to_original"]) // 2] for x in perm)
        p(f"Permutation test on {len(perm)} v1.1 items (options reversed, probabilities mapped back to the original "
          f"letters): the argmax stays on the same option *text* in {same}/{len(perm)}; the key stays in the lower half "
          f"of the option probabilities in {lowkey}/{len(perm)}. The low score follows the content of the key, not its "
          "letter or position.")
    p("")
    print("\n".join(P))


if __name__ == "__main__":
    main()
