"""Write the ```sketch fence for the S-EMBER_7291 coverage timeline from coverage.json.

Only time ranges and timestamps from the traces (no frames). Usage: python3 -I make_coverage_sketch.py coverage.json > out.md
"""
import json, sys

d = json.load(open(sys.argv[1]))
it = next(i for i in d["items"] if i["item_id"] == "S-EMBER_7291")
A = {a["model"]: a for a in it["agents"]}
argon, astra = A["gemini-4-argon"], A["gpt-6-astra"]
DUR = it["duration_s"]
events = [e["t"] for e in astra["events_reported"]]              # 129, 132, 134, 232, 249 (astra's own words)
counted = [(e["t"], e["t_end"]) for e in argon["events_reported"]]  # argon's 3 intervals
missed = [t for t in events if not any(a - 1 <= t <= b + 1 for a, b in counted)]
assert missed == [134, 232], missed
win = lambda a, f: [(r["t0"], r["t1"], r["fps"]) for r in a["ranges"] if r["fps"] is not None]
AW, SW = win(argon, None), win(astra, None)
assert AW[0][:2] == (0, 319) and SW[0][:2] == (0, 319)

def f(x):
    return f"{x:.1f}".rstrip("0").rstrip(".")

def band(out, x, rows, y, t0, t1, fps, scale, ox, cls):
    x0, x1 = ox + max(t0, x[0]) * 0 + (max(t0, x[0]) - x[0]) * scale, ox + (min(t1, x[1]) - x[0]) * scale
    if x1 <= x0:
        return
    h = 6 if fps <= 1 else 14
    out.append(f'<rect x="{f(x0)}" y="{f(y - h / 2)}" width="{f(max(x1 - x0, 1.5))}" height="{h}" rx="2" fill="var(--{cls})"/>')

def timeline(out, ox, ow, rng, y_ev, y_ar, y_as, label_x, labels, ticks, num=True):
    scale = ow / (rng[1] - rng[0])
    X = lambda t: ox + (t - rng[0]) * scale
    out.append(f'<line class="edge" x1="{f(ox)}" y1="{y_as + 24}" x2="{f(ox + ow)}" y2="{y_as + 24}"/>')
    for t in ticks:
        out.append(f'<line class="edge" x1="{f(X(t))}" y1="{y_as + 21}" x2="{f(X(t))}" y2="{y_as + 27}"/><text class="small mid" x="{f(X(t))}" y="{y_as + 38}">{t} s</text>')
    if labels:
        out.append(f'<text class="small" x="{label_x}" y="{y_ev}">{labels[0]}</text>')
        out.append(f'<text x="{label_x}" y="{y_ar}">{labels[1]}</text>')
        out.append(f'<text x="{label_x}" y="{y_as}">{labels[2]}</text>')
    for name, a, y in (("argon", AW, y_ar), ("astra", SW, y_as)):
        out.append(f'<line x1="{f(ox)}" y1="{y}" x2="{f(ox + ow)}" y2="{y}" stroke="var(--line)"/>')
        for t0, t1, fps in a:
            band(out, rng, None, y, t0, t1, fps, scale, ox, "stone" if fps <= 1 else "blue")
    for a, b in counted:
        if rng[0] <= a <= rng[1]:
            out.append(f'<rect x="{f(X(a))}" y="{y_ar - 9}" width="{f(max((b - a) * scale, 3))}" height="18" rx="2" fill="none" stroke="var(--ink)" stroke-width="1.5"/>')
    for k, t in enumerate(events, 1):
        if not rng[0] <= t <= rng[1]:
            continue
        cls = "clay" if t in missed else "teal"
        out.append(f'<line x1="{f(X(t))}" y1="{y_ev + (8 if num else 4)}" x2="{f(X(t))}" y2="{y_as + 8}" stroke="var(--{cls})" stroke-dasharray="2 3"/>')
        if num:
            out.append(f'<g class="node {cls}"><circle cx="{f(X(t))}" cy="{y_ev}" r="8"/><text class="mid" x="{f(X(t))}" y="{y_ev}" font-size="11">{k}</text></g>')
        else:
            out.append(f'<circle cx="{f(X(t))}" cy="{y_ev}" r="3.5" fill="var(--{cls})"/>')

def svg(W, narrow):
    out = []
    if not narrow:
        H = 392
        out.append(f'<svg viewBox="0 0 {W} {H}">')
        out.append(f'<rect class="panel" x="0" y="0" width="{W}" height="{H}" rx="12"/>')
        out.append('<text class="caps" x="24" y="30">整段视频 · 319 s</text>')
        timeline(out, 176, 504, (0, DUR), 62, 92, 120, 24, ["回答正确者报告的指向", "Gemini 4 Argon", "GPT-6-Astra"], [0, 100, 200, 300], num=False)
        out.append('<text class="small" x="24" y="106">答 3 次</text><text class="small" x="24" y="134">答 5 次（正确）</text>')
        for k, (r, x0) in enumerate((((120, 145), 24), ((225, 258), 364))):
            out.append(f'<text class="caps" x="{x0}" y="210">放大 {r[0]}–{r[1]} s</text>')
            timeline(out, x0 + 56, 260, r, 238, 270, 300, x0, None, [r[0], (r[0] + r[1]) // 2 if k == 0 else 240, r[1]])
            out.append(f'<text class="small" x="{x0}" y="270">Argon</text><text class="small" x="{x0}" y="300">Astra</text>')
        out.append('<rect x="24" y="356" width="18" height="6" rx="2" fill="var(--stone)"/><text class="small" x="48" y="359">≤1 帧/秒的整段浏览</text>')
        out.append('<rect x="176" y="352" width="18" height="14" rx="2" fill="var(--blue)"/><text class="small" x="200" y="359">2–6 帧/秒的细看窗口</text>')
        out.append('<rect x="340" y="351" width="16" height="16" rx="2" fill="none" stroke="var(--ink)" stroke-width="1.5"/><text class="small" x="362" y="359">Argon 数到的指向</text>')
        out.append('<circle cx="480" cy="359" r="6" fill="var(--clay)"/><text class="small" x="492" y="359">落在 Argon 细看过的窗口里，却没数到</text>')
    else:
        H = 560
        out.append(f'<svg viewBox="0 0 {W} {H}">')
        out.append(f'<rect class="panel" x="0" y="0" width="{W}" height="{H}" rx="12"/>')
        out.append('<text class="caps" x="16" y="28">整段视频 · 319 s</text>')
        timeline(out, 96, 224, (0, DUR), 56, 86, 114, 16, ["正确者报告", "Argon", "Astra"], [0, 150, 300], num=False)
        out.append('<text class="small" x="16" y="100">答 3 次</text><text class="small" x="16" y="128">答 5 次 ✓</text>')
        for k, (r, y0) in enumerate((((120, 145), 196), ((225, 258), 340))):
            out.append(f'<text class="caps" x="16" y="{y0}">放大 {r[0]}–{r[1]} s</text>')
            timeline(out, 72, 248, r, y0 + 30, y0 + 62, y0 + 92, 16, None, [r[0], (r[0] + r[1]) // 2 if k == 0 else 240, r[1]])
            out.append(f'<text class="small" x="16" y="{y0 + 62}">Argon</text><text class="small" x="16" y="{y0 + 92}">Astra</text>')
        out.append('<rect x="16" y="484" width="18" height="6" rx="2" fill="var(--stone)"/><text class="small" x="40" y="487">≤1 帧/秒浏览</text>')
        out.append('<rect x="170" y="480" width="18" height="14" rx="2" fill="var(--blue)"/><text class="small" x="194" y="487">2–6 帧/秒细看</text>')
        out.append('<rect x="16" y="508" width="16" height="16" rx="2" fill="none" stroke="var(--ink)" stroke-width="1.5"/><text class="small" x="40" y="516">Argon 数到的指向</text>')
        out.append('<circle cx="24" cy="542" r="6" fill="var(--clay)"/><text class="small" x="40" y="542">在 Argon 细看的窗口里，没数到</text>')
    out.append("</svg>")
    return "\n".join(out)

print(svg(704, False))
print(svg(340, True))
