"""One chart theme for the whole repository.

Two rules drive everything here.

Colour is never the only channel carrying meaning. The palette is Okabe-Ito,
which stays distinguishable under the three common forms of colour blindness,
and MARKERS / DASHES are exposed so a second channel can always be added.

A chart title states the finding, not the axis names. `title()` takes a headline
and an optional quieter subtitle underneath it, which is where the setup goes.

Author: Elyes Lounissi
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito, reordered so the first three are also the furthest apart in
# greyscale. Yellow is last because it disappears on a white page.
PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # rose
    "#56B4E9",  # sky
    "#8C6D31",  # brown
    "#F0E442",  # yellow
]

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
DASHES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (7, 2))]

INK = "#1a1a1a"      # primary text
MUTED = "#6b6b6b"    # secondary text, subtitles, annotations
RULE = "#d8d8d8"     # gridlines and spines
HIGHLIGHT = "#D55E00"  # the one series the reader should look at
NEUTRAL = "#b0b0b0"    # every other series when one is highlighted


def use() -> None:
    """Apply the repo theme. Call once, near the top of a notebook."""
    mpl.rcParams.update({
        "figure.figsize": (7.2, 4.4),
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",

        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,

        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": RULE,
        "xtick.color": MUTED,
        "ytick.color": MUTED,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": RULE,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.9,
        "axes.axisbelow": True,

        "lines.linewidth": 1.9,
        "lines.markersize": 5.5,
        "patch.edgecolor": "white",
        "patch.linewidth": 0.7,

        "legend.frameon": False,
        "legend.handlelength": 1.8,

        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


def title(ax, headline: str, sub: str | None = None) -> None:
    """Left-aligned headline stating the finding, with an optional quiet subtitle.

    The subtitle sits inside the axes title block rather than in a text box, so
    it survives `bbox_inches="tight"` without manual padding.
    """
    if sub:
        ax.set_title(headline, loc="left", fontsize=12, color=INK, pad=22)
        ax.annotate(
            sub, xy=(0, 1), xycoords="axes fraction",
            xytext=(0, 8), textcoords="offset points",
            ha="left", va="bottom", fontsize=9.5, color=MUTED,
        )
    else:
        ax.set_title(headline, loc="left", fontsize=12, color=INK, pad=10)


def note(ax, text: str) -> None:
    """A source or caveat line under the x-axis, in the muted colour."""
    ax.annotate(
        text, xy=(0, 0), xycoords="axes fraction",
        xytext=(0, -42), textcoords="offset points",
        ha="left", va="top", fontsize=8.5, color=MUTED, wrap=True,
    )


def spot(n: int, focus: int) -> list[str]:
    """Colours for `n` series where only `focus` should draw the eye.

    Greying out the rest is more honest than giving eight series eight loud
    colours and letting the reader guess which comparison matters.
    """
    return [HIGHLIGHT if i == focus else NEUTRAL for i in range(n)]


def save(fig, path, *, close: bool = False) -> None:
    """Write a figure to `path`, creating parent folders as needed.

    The figure is left open by default so that a notebook cell still renders it
    inline after saving. Scripts that produce many figures should pass
    close=True.
    """
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if close:
        plt.close(fig)
