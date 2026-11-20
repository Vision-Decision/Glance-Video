"""Print the echarts / frames fences for the Glance post from the raw output files.

python3 -I make_post_figs.py  (run from outputs/blog). Every number comes from:
  ../ablation/main_table.json, ../ablation/halluc_vs_negratio.json, glance_bbb.json
"""
import json

def r1(x):
    return round(x * 100, 1)

# --- dumbbell 1: refusal on present targets, greedy vs sequence MAP
T = {r["run"]: r for r in json.load(open("../ablation/main_table.json"))}
rows = [("AR 数字", "ar_digits_neg25"), ("AR 层级", "ar_hier_neg25"), ("AR 原子 1000", "ar_atomic1000_neg25"),
        ("AR 原子 100", "ar_atomic100_neg25"), ("AR 数字 · 负 5%", "ar_digits_neg05"), ("门控", "ours_neg25"), ("门控 · 负 5%", "ours_neg05")]
cats = [c for c, _ in rows]
greedy = [r1(T[k]["refusal_present_val"]) for _, k in rows]
smap = [r1(T[k]["refusal_seqmap_val"]) if "refusal_seqmap_val" in T[k] else None for _, k in rows]
lo = [min(g, s) if s is not None else g for g, s in zip(greedy, smap)]
span = [round(abs(s - g), 1) if s is not None else 0 for g, s in zip(greedy, smap)]
opt = {
    "tooltip": {"trigger": "item"},
    "grid": {"left": 112, "right": 56, "top": 16, "bottom": 40},
    "xAxis": {"type": "value", "max": 45, "name": "在场却说“不在”", "nameLocation": "middle", "nameGap": 26, "axisLabel": {"formatter": "{value}%"}},
    "yAxis": {"type": "category", "data": cats, "inverse": True, "axisTick": {"show": False}},
    "series": [
        {"type": "bar", "stack": "d", "silent": True, "itemStyle": {"color": "transparent"}, "data": lo, "tooltip": {"show": False}},
        {"type": "bar", "stack": "d", "silent": True, "barWidth": 3, "itemStyle": {"color": "var(--line)"}, "data": span, "tooltip": {"show": False}},
        {"name": "贪心", "type": "scatter", "symbolSize": 11, "itemStyle": {"color": "var(--stone)"}, "z": 3,
         "label": {"show": True, "position": "bottom", "distance": 4, "formatter": "{@[0]}%", "color": "var(--muted)", "fontSize": 10, "textBorderColor": "transparent"},
         "data": [[g, i] for i, g in enumerate(greedy)], "tooltip": {"formatter": "{a}：{@[0]}%"}},
        {"name": "序列 MAP", "type": "scatter", "symbolSize": 11, "itemStyle": {"color": "var(--clay)"}, "z": 4,
         "label": {"show": True, "position": "right", "formatter": "{@[0]}%", "color": "var(--ink)", "textBorderColor": "transparent"},
         "data": [[s, i] for i, s in enumerate(smap) if s is not None], "tooltip": {"formatter": "{a}：{@[0]}%"}},
    ],
}
print("DUMBBELL1", json.dumps(opt, ensure_ascii=False, indent=None))
print("  values", list(zip(cats, greedy, smap)))

# --- dumbbell 2: hallucination 25% -> 5% negatives, plus prior correction
H = json.load(open("../ablation/halluc_vs_negratio.json"))
rows2 = [("门控", "ours/gate"), ("AR 数字 · 阈值", "ar_digits/thresh"), ("AR 数字 · 贪心", "ar_digits/greedy")]
v25 = [round(H[k]["halluc_pct"][1], 1) for _, k in rows2]
v05 = [round(H[k]["halluc_pct"][0], 1) for _, k in rows2]
pc = [round(H[k]["halluc_pct_5pct_prior_corrected"], 1) if H[k]["halluc_pct_5pct_prior_corrected"] is not None else None for _, k in rows2]
opt2 = {
    "tooltip": {"trigger": "item"},
    "legend": {"data": ["负样本 25%", "负样本 5%", "5% + 先验修正"]},
    "grid": {"left": 112, "right": 56, "top": 40, "bottom": 40},
    "xAxis": {"type": "value", "min": 15, "max": 40, "name": "无目标表达式上画了框", "nameLocation": "middle", "nameGap": 26, "axisLabel": {"formatter": "{value}%"}},
    "yAxis": {"type": "category", "data": [c for c, _ in rows2], "inverse": True, "axisTick": {"show": False}},
    "series": [
        {"type": "bar", "stack": "d", "silent": True, "itemStyle": {"color": "transparent"}, "data": v25, "tooltip": {"show": False}},
        {"type": "bar", "stack": "d", "silent": True, "barWidth": 3, "itemStyle": {"color": "var(--line)"}, "data": [round(b - a, 1) for a, b in zip(v25, v05)], "tooltip": {"show": False}},
        {"name": "负样本 25%", "type": "scatter", "symbolSize": 11, "itemStyle": {"color": "var(--stone)"}, "z": 3,
         "label": {"show": True, "position": "left", "formatter": "{@[0]}%", "color": "var(--muted)", "textBorderColor": "transparent"},
         "data": [[v, i] for i, v in enumerate(v25)], "tooltip": {"formatter": "{a}：{@[0]}%"}},
        {"name": "负样本 5%", "type": "scatter", "symbolSize": 11, "itemStyle": {"color": "var(--clay)"}, "z": 4,
         "label": {"show": True, "position": "right", "formatter": "{@[0]}%", "color": "var(--ink)", "textBorderColor": "transparent"},
         "data": [[v, i] for i, v in enumerate(v05)], "tooltip": {"formatter": "{a}：{@[0]}%"}},
        {"name": "5% + 先验修正", "type": "scatter", "symbolSize": 10, "itemStyle": {"color": "var(--paper)", "borderColor": "var(--clay)", "borderWidth": 2}, "z": 5,
         "label": {"show": True, "position": "top", "formatter": "{@[0]}%", "color": "var(--muted)", "fontSize": 10, "textBorderColor": "transparent"},
         "data": [[v, i] for i, v in enumerate(pc) if v is not None], "tooltip": {"formatter": "{a}：{@[0]}%"}},
    ],
}
print("DUMBBELL2", json.dumps(opt2, ensure_ascii=False))
print("  values", list(zip(rows2, v25, v05, pc)))

# --- P(t) on Big Buck Bunny
G = json.load(open("glance_bbb.json"))
ts = [r["t"] for r in G["rows"]]
rab = [round(r["results"][0]["present_prob"], 3) for r in G["rows"]]
but = [round(r["results"][3]["p_yes"], 3) for r in G["rows"]]

def intervals(ps):
    out, cur = [], None
    for t, p in zip(ts, ps):
        if p >= 0.5:
            cur = [t, t] if cur is None else [cur[0], t]
        elif cur:
            out.append(cur); cur = None
    if cur:
        out.append(cur)
    return out

def areas(ps):
    # each sampled frame stands for +-4 s around it
    return [[{"xAxis": max(a - 4, 0)}, {"xAxis": min(b + 4, 480)}] for a, b in intervals(ps)]

def ser(name, ps, xi, color, wash):
    return {"name": name, "type": "line", "xAxisIndex": xi, "yAxisIndex": xi, "showSymbol": True, "symbolSize": 4,
            "lineStyle": {"width": 1.5}, "itemStyle": {"color": color}, "data": [[t, p] for t, p in zip(ts, ps)],
            "markLine": {"silent": True, "symbol": "none", "lineStyle": {"color": "var(--muted)", "type": "dashed"},
                         "label": {"formatter": "阈值 0.5", "position": "insideEndTop", "color": "var(--muted)"}, "data": [{"yAxis": 0.5}]},
            "markArea": {"silent": True, "itemStyle": {"color": wash}, "data": areas(ps)}}

ax = lambda i, name: {"type": "value", "gridIndex": i, "min": 0, "max": 480, "interval": 60, "name": name, "nameLocation": "middle", "nameGap": 24,
                      "axisLabel": {"formatter": "{value} s"}}
opt3 = {
    "tooltip": {"trigger": "axis"},
    "grid": [{"left": 44, "right": 16, "top": 28, "height": 120}, {"left": 44, "right": 16, "top": 232, "height": 120}],
    "xAxis": [ax(0, ""), ax(1, "视频时间")],
    "yAxis": [{"type": "value", "gridIndex": 0, "min": 0, "max": 1, "interval": 0.5, "name": "P(present)：the big white rabbit", "nameLocation": "end", "nameTextStyle": {"align": "left"}},
              {"type": "value", "gridIndex": 1, "min": 0, "max": 1, "interval": 0.5, "name": "P(yes)：Is there a butterfly in this frame?", "nameLocation": "end", "nameTextStyle": {"align": "left"}}],
    "series": [ser("兔子 P(present)", rab, 0, "var(--stone)", "var(--wash)"), ser("蝴蝶 P(yes)", but, 1, "var(--clay)", "var(--clay-wash)")],
}
print("PT", json.dumps(opt3, ensure_ascii=False))
print("  rabbit intervals", intervals(rab), "butterfly", intervals(but))
