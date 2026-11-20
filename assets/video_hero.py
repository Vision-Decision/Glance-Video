"""glance-video hero: the vdmscan tool inside an agent loop (schematic; tool contract from
video/vdm/gv/TOOL_scan.md: 2-s clips, at most 600 per video, PRESENT calibrated to ~10% false alarms)."""
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
from figstyle import render

def box(ax, x, y, w, h, fc, ec, lw=1.2, r=0.8, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))

def arrow(ax, p0, p1, t, col=None, lw=1.4, cs="arc3"):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=col or t["muted"], lw=lw,
                mutation_scale=12, shrinkA=0, shrinkB=0, connectionstyle=cs))

def draw(t, mode):
    fig = plt.figure(figsize=(12.8, 4.9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 128); ax.set_ylim(1.5, 54); ax.axis("off")
    # agent
    box(ax, 2, 18, 24, 20, t["card"], t["cardline"])
    ax.text(14, 33.5, "Video agent", ha="center", fontsize=13, weight="bold", color=t["ink"])
    ax.text(14, 30.3, "any LLM planner", ha="center", fontsize=9.5, color=t["ink2"])
    ax.text(14, 25.2, "breaks the question into\nshort visual events,\nthen reasons over what\nthe tool found",
            ha="center", va="center", fontsize=9, color=t["ink2"], linespacing=1.3)
    # call
    arrow(ax, (26.4, 34), (40, 44), t, col=t["blue"], cs="arc3,rad=-0.15")
    box(ax, 30, 45.2, 52, 4.6, t["blue_t"], t["blue"], lw=1, r=0.6)
    ax.text(31.5, 47.5, 'vdmscan find video.mp4 "picks up a red book" --k 6', fontsize=9.6, family="DejaVu Sans Mono",
            color=t["ink"], va="center")
    # scan server
    box(ax, 40, 12, 52, 30.5, t["card"], t["cardline"], r=1.0, z=1)
    ax.text(42, 39.5, "Scan server: Glance-Video (4B)", fontsize=12, weight="bold", color=t["blue"], va="center")
    # clips
    n = 20
    for i in range(n):
        x = 42 + i * 2.45
        box(ax, x, 31.0, 2.1, 4.6, t["gray_t"], t["cardline"], lw=0.7, r=0.3)
    ax.text(42, 29.6, "video split into 2-s clips (up to 600), each encoded once and cached",
            fontsize=9, color=t["ink2"], va="top")
    ax.text(42, 24.6, "one forward pass asks the phrase of every cached clip:", fontsize=9, color=t["ink2"], va="center")
    hi = {6, 7, 14}
    for i in range(n):
        x = 42 + i * 2.45
        h = 7.2 if i in hi else (1.2 + 0.9 * ((i * 7) % 5) / 4)
        box(ax, x, 14.2, 2.1, h, t["blue"] if i in hi else t["blue_t"], t["blue"] if i in hi else t["cardline"],
            lw=0.7, r=0.3)
    ax.text(90.5, 13.0, "per-clip P(event), schematic", ha="right", fontsize=8, color=t["muted"], va="top")
    # result
    arrow(ax, (92.4, 27), (98.6, 27), t, col=t["blue"])
    box(ax, 99, 12, 27, 30.5, t["blue_t"], t["blue"])
    ax.text(101, 39.5, "Tool returns", fontsize=12, weight="bold", color=t["ink"], va="center")
    items = [("top-k moments", "spread in time, with probabilities"),
             ("a verdict", "PRESENT: ~10% false alarms\non videos without the event"),
             ("one contact sheet", "4 frames per moment,\nin time order")]
    for i, (k, v) in enumerate(items):
        y = 34.5 - i * 7.6
        ax.text(101, y, k, fontsize=9.8, weight="bold", color=t["blue"], va="top")
        ax.text(101, y - 2.4, v, fontsize=8.8, color=t["ink"], va="top", linespacing=1.25)
    arrow(ax, (112, 11.6), (16, 17.6), t, col=t["blue"], cs="arc3,rad=-0.12")
    ax.text(64, 4.2, "the agent looks only at the returned moments, then asks the next question or answers",
            ha="center", fontsize=9.5, color=t["ink2"])
    return fig

if __name__ == "__main__":
    render("video-scan-loop", draw)
