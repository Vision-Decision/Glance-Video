"""The set-level PRESENT verdict holds its false-alarm rate as scans get longer; a fixed 0.5 threshold does not.
Source: video/gate/results/gate_stats.json (held-out Charades-STA test videos, alpha = 0.1):
ever.<N>.false_alarm, naive_p05_false_alarm, detection; N = natural length (median 29 clips) and concatenated scans."""
import json, os
import matplotlib.pyplot as plt
from figstyle import render

P = os.environ.get("GATE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "video", "gate", "results", "gate_stats.json"))
G = json.load(open(P))
ev = G["ever"]
KEYS = [("natural", ev["natural"]["median_clips"])] + [(k, int(k)) for k in ("32", "64", "128", "256", "512", "900")]

def draw(t, mode):
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    fig.subplots_adjust(left=0.08, right=0.8, top=0.82, bottom=0.15)
    xs = [n for _, n in KEYS]
    naive = [100 * ev[k]["naive_p05_false_alarm"] for k, _ in KEYS]
    cal = [100 * ev[k]["false_alarm"] for k, _ in KEYS]
    det = [100 * ev[k]["detection"] for k, _ in KEYS]
    ax.axhline(10, color=t["ink2"], lw=1)
    ax.text(xs[0] * 0.93, 12.5, "target: 10% false alarms", fontsize=9, color=t["ink2"])
    for ys, c, lab in [(naive, t["orange"], "false alarms, fixed P > 0.5 threshold"),
                       (cal, t["blue"], "false alarms, calibrated PRESENT verdict"),
                       (det, t["aqua"], "detection rate, calibrated verdict")]:
        ax.plot(xs, ys, color=c, lw=2.2, marker="o", ms=7, mec="none", solid_capstyle="round")
        ax.text(xs[-1] * 1.08, ys[-1], f"{lab}\n{ys[-1]:.0f}% at {xs[-1]} clips", va="center", fontsize=9.5, color=t["ink"])
    ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs]); ax.minorticks_off()
    ax.set_ylim(0, 105); ax.set_ylabel("% of scans", fontsize=10)
    ax.set_xlabel("clips per scan (first point: single videos, median length; vdmscan caps a scan at 600 clips)", fontsize=10, color=t["ink2"])
    ax.grid(axis="y", color=t["grid"], lw=1); ax.set_axisbelow(True)
    ax.set_title("A fixed threshold cries wolf on long videos; the calibrated verdict stays near 10%", loc="left",
                 fontsize=12.5, weight="bold", color=t["ink"], pad=24)
    ax.text(0, 1.03, "held-out Charades-STA test videos, concatenated into longer scans; threshold fitted per length on separate videos", transform=ax.transAxes,
            fontsize=9.5, color=t["ink2"])
    return fig

if __name__ == "__main__":
    render("video-calibration", draw)
