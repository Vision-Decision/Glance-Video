"""Glance-Video vs zero-shot Qwen3.5-4B (same native video prefix, LM readout on the label tokens).
Sources (copied into this repo): outputs/gv0/full_typed.json and base_typed.json (raw.<task>.acc, raw.<task>.chance),
outputs/gv0/full_repcount.json and base_repcount.json (pred_argmax.MAE)."""
import json, os
import matplotlib.pyplot as plt
from figstyle import render

R = os.environ.get("GV0", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "gv0"))
F = json.load(open(f"{R}/full_typed.json"))["raw"]; B = json.load(open(f"{R}/base_typed.json"))["raw"]
FR = json.load(open(f"{R}/full_repcount.json")); BR = json.load(open(f"{R}/base_repcount.json"))
TASKS = [("count", "Repetition count, Countix val (5-way)"),
         ("event_choice", "Which event happens, Charades (4-way)"),
         ("camera", "Camera motion, Kinetics-700 (choice)"),
         ("event_order", "Event order, Charades (2-way)"),
         ("aot", "Arrow of time, Kinetics-700 (2-way)")]

def draw(t, mode):
    fig = plt.figure(figsize=(12.8, 4.6))
    ax = fig.add_axes([0.27, 0.15, 0.43, 0.64]); ax2 = fig.add_axes([0.79, 0.15, 0.17, 0.62])
    n = len(TASKS)
    for i, (k, lab) in enumerate(TASKS):
        y = n - 1 - i
        g, b, ch = 100 * F[k]["acc"], 100 * B[k]["acc"], 100 * F[k]["chance"]
        ax.barh(y + 0.19, g, height=0.36, color=t["blue"])
        ax.barh(y - 0.19, b, height=0.36, color=t["orange"])
        ax.plot([ch, ch], [y - 0.42, y + 0.42], color=t["ink"], lw=1.6, zorder=4)
        ax.text(g + 1, y + 0.19, f"{g:.1f}", va="center", fontsize=9.5, weight="bold", color=t["ink"])
        ax.text(b + 1, y - 0.19, f"{b:.1f}", va="center", fontsize=9, color=t["ink2"])
        ax.text(-1.5, y - 0.48, f"n = {F[k]['n']:,}", ha="right", va="center", fontsize=8, color=t["muted"])
    ax.set_yticks(range(n)); ax.set_yticklabels([x[1] for x in TASKS][::-1], fontsize=10, color=t["ink"])
    ax.set_xlim(0, 100); ax.grid(axis="x", color=t["grid"], lw=1); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("accuracy (%);  vertical tick = chance level", fontsize=9.5, color=t["ink2"])
    ax.set_title("Training lifts counting from 29% to 93%; arrow of time stays near chance", loc="left", fontsize=12, weight="bold", color=t["ink"], pad=24)
    ax.text(0, 1.025, "held-out clips; Glance-Video vs the base model on the same video prefix", transform=ax.transAxes, fontsize=9, color=t["ink2"])
    g, b = FR["pred_argmax"]["MAE"], BR["pred_argmax"]["MAE"]
    ax2.bar([0], [g], width=0.6, color=t["blue"]); ax2.bar([1], [b], width=0.6, color=t["orange"])
    ax2.text(0, g + 0.015, f"{g:.3f}", ha="center", fontsize=10, weight="bold", color=t["ink"])
    ax2.text(1, b + 0.015, f"{b:.3f}", ha="center", fontsize=9.5, color=t["ink2"])
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Glance-\nVideo", "Qwen3.5-4B\nbase"], fontsize=9, color=t["ink"])
    ax2.set_ylim(0, 0.62); ax2.grid(axis="y", color=t["grid"], lw=1); ax2.set_axisbelow(True)
    ax2.tick_params(axis="y", labelsize=8.5)
    ax2.set_title(f"RepCount-A counting error\nrelative MAE, lower is better\nn = {FR['n']}", loc="left", fontsize=10.5,
                  weight="bold", color=t["ink"], pad=10)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=t["blue"], label="Glance-Video"), Patch(color=t["orange"], label="Qwen3.5-4B base, zero-shot")],
               loc="upper left", bbox_to_anchor=(0.27, 1.0), ncol=2, frameon=False, fontsize=10, labelcolor=t["ink"])
    return fig

if __name__ == "__main__":
    render("video-results", draw)
