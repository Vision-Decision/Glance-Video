"""Shared figure style for the Vision-Decision READMEs.

Every figure is drawn twice, once per GitHub theme, onto a transparent canvas, and
saved as <name>-light.svg / <name>-dark.svg (text converted to paths, so it renders
the same everywhere). README embeds them with <picture>.
Palette: the validated dataviz reference palette (categorical slots 1-3 + inks).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THEMES = {
    "light": dict(
        ink="#0b0b0b", ink2="#52514e", muted="#6f6e69", grid="#e1e0d9", axis="#c3c2b7",
        blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", violet="#6250d6", red="#d03b3b",
        gray="#898781", card="#f6f6f3", cardline="#d9d8d1", onfill="#ffffff",
        blue_t="#e3eefb", orange_t="#fde9e0", aqua_t="#dcf3ea", violet_t="#e9e6fa", gray_t="#efeeea",
    ),
    "dark": dict(
        ink="#f0efec", ink2="#c3c2b7", muted="#9a9890", grid="#2c2c2a", axis="#45453f",
        blue="#3987e5", orange="#d95926", aqua="#199e70", violet="#9085e9", red="#e66767",
        gray="#898781", card="#1a1c20", cardline="#3a3b3f", onfill="#ffffff",
        blue_t="#14263d", orange_t="#3a1d12", aqua_t="#0f2e23", violet_t="#25213f", gray_t="#222326",
    ),
}

def setup(t):
    plt.rcParams.update({
        "font.family": ["Noto Sans", "DejaVu Sans"], "font.size": 11, "svg.fonttype": "path",
        "text.color": t["ink"], "axes.labelcolor": t["ink2"], "axes.edgecolor": t["axis"],
        "xtick.color": t["ink2"], "ytick.color": t["ink2"], "axes.linewidth": 1.0,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.transparent": True,
    })

def render(name, draw, outdir="out", **kw):
    """draw(fig_factory, t) must return a Figure."""
    import os
    os.makedirs(outdir, exist_ok=True)
    for mode, t in THEMES.items():
        setup(t)
        fig = draw(t, mode, **kw)
        fig.savefig(f"{outdir}/{name}-{mode}.svg", transparent=True, metadata={"Date": None})
        fig.savefig(f"{outdir}/{name}-{mode}.png", transparent=False, dpi=110,
                    facecolor="#ffffff" if mode == "light" else "#0d1117")
        plt.close(fig)
