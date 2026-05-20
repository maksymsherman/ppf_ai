# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib>=3.8",
#   "numpy>=1.26",
# ]
# ///
"""Draw a schematic PPF for AI-assisted coding."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


CLEAN_BLACK = "#333333"
CLEAN_DARK_GRAY = "#555555"
CLEAN_MEDIUM_GRAY = "#888888"
CLEAN_LIGHT_GRAY = "#cccccc"
CLEAN_REF_GRAY = "#d0d0d0"

TOL_INDIGO = "#332288"
TOL_ROSE = "#CC6677"
TOL_GREEN = "#117733"

CLEAN_FONT_SIZE = 11
CLEAN_TITLE_SIZE = 13
CLEAN_LABEL_SIZE = 10
CLEAN_SMALL_SIZE = 9

X_MAX = 15.5
Y_MAX = 10.0
POST_CEILING = 9.45
POST_FLOOR = 2.05
POST_CENTER = 7.1
POST_SCALE = 0.72


def configure_matplotlib() -> None:
    """Apply a sparse publication-style matplotlib configuration."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": CLEAN_FONT_SIZE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": CLEAN_TITLE_SIZE,
            "axes.titleweight": "normal",
            "axes.labelsize": CLEAN_FONT_SIZE,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.size": 0,
            "ytick.minor.size": 0,
            "axes.grid": False,
            "grid.alpha": 0,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 200,
            "figure.dpi": 120,
        }
    )


def pre_ai_frontier() -> tuple[np.ndarray, np.ndarray]:
    """Return a conventional pre-AI concave tradeoff curve."""
    t = np.linspace(0, 1, 260)
    x = 0.8 + 4.6 * t
    y = 9.2 - 6.8 * (t**1.85)
    return x, y


def post_ai_quality(x: float | np.ndarray) -> float | np.ndarray:
    """Return the post-AI frontier quality at a given output index."""
    x_array = np.asarray(x)
    quality = POST_FLOOR + (POST_CEILING - POST_FLOOR) / (
        1 + np.exp((x_array - POST_CENTER) / POST_SCALE)
    )
    if np.isscalar(x):
        return float(quality)
    return quality


def post_ai_volume_at_quality(quality: float) -> float:
    """Invert the post-AI frontier for a quality value above the floor."""
    ratio = (POST_CEILING - POST_FLOOR) / (quality - POST_FLOOR) - 1
    return POST_CENTER + POST_SCALE * np.log(ratio)


def apply_index_frame(ax: plt.Axes) -> None:
    """Apply bounded axes for the schematic index ranges."""
    ax.set_xlim(-0.2, X_MAX + 1.0)
    ax.set_ylim(-0.1, Y_MAX + 0.55)
    ax.spines["bottom"].set_bounds(0, X_MAX)
    ax.spines["left"].set_bounds(0, Y_MAX)
    ax.xaxis.set_major_locator(ticker.FixedLocator([0, 5, 10, 15]))
    ax.yaxis.set_major_locator(ticker.FixedLocator([0, 2.5, 5, 7.5, 10]))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax.tick_params(colors=CLEAN_BLACK, labelsize=CLEAN_SMALL_SIZE)


def draw_move(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    label: str,
    label_xy: tuple[float, float],
    *,
    ha: str = "left",
    va: str = "center",
    lw: float = 1.4,
) -> None:
    """Draw one operating move from the pre-AI baseline."""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "lw": lw,
            "mutation_scale": 12,
            "shrinkA": 5,
            "shrinkB": 5,
        },
        zorder=3,
    )
    ax.scatter(
        [end[0]],
        [end[1]],
        s=34,
        facecolor="white",
        edgecolor=color,
        linewidth=1.4,
        zorder=5,
    )
    ax.text(
        label_xy[0],
        label_xy[1],
        label,
        ha=ha,
        va=va,
        fontsize=CLEAN_LABEL_SIZE,
        color=CLEAN_BLACK,
        linespacing=1.15,
    )


def build_chart() -> plt.Figure:
    """Build and return the PPF figure."""
    configure_matplotlib()

    pre_x, pre_y = pre_ai_frontier()
    post_x = np.linspace(0.7, X_MAX, 520)
    post_y = post_ai_quality(post_x)

    baseline_x = 3.2
    baseline_y = float(np.interp(baseline_x, pre_x, pre_y))
    baseline = (baseline_x, baseline_y)

    careful = (baseline_x, post_ai_quality(baseline_x))
    disciplined = (post_ai_volume_at_quality(baseline_y), baseline_y)
    vibe = (13.6, post_ai_quality(13.6))

    fig, ax = plt.subplots(figsize=(9, 5.7))

    ax.plot(
        pre_x,
        pre_y,
        color=CLEAN_MEDIUM_GRAY,
        linestyle=(0, (5, 3)),
        linewidth=1.3,
        zorder=1,
    )
    ax.plot(post_x, post_y, color=CLEAN_BLACK, linewidth=1.8, zorder=2)
    ax.hlines(
        POST_FLOOR,
        8.9,
        X_MAX,
        color=CLEAN_REF_GRAY,
        linestyle=(0, (2, 3)),
        linewidth=0.9,
        zorder=0,
    )

    ax.scatter(
        [baseline_x],
        [baseline_y],
        s=38,
        facecolor="white",
        edgecolor=CLEAN_DARK_GRAY,
        linewidth=1.2,
        zorder=5,
    )

    draw_move(
        ax,
        baseline,
        careful,
        TOL_GREEN,
        "A. Careful adopter\nsame volume, higher quality",
        (3.75, 10.2),
        va="top",
    )
    draw_move(
        ax,
        baseline,
        disciplined,
        TOL_INDIGO,
        "B. Disciplined volume\nmore volume, same quality",
        (6.75, 7.55),
        va="bottom",
    )
    draw_move(
        ax,
        baseline,
        vibe,
        TOL_ROSE,
        "C. Vibe coding\nmuch more output, lower quality",
        (12.05, 3.38),
        lw=1.8,
    )

    ax.text(
        baseline_x - 0.15,
        baseline_y - 0.45,
        "Pre-AI\nbaseline",
        ha="right",
        va="top",
        fontsize=CLEAN_SMALL_SIZE,
        color=CLEAN_BLACK,
        linespacing=1.1,
    )
    ax.text(
        1.0,
        8.85,
        "Pre-AI frontier",
        ha="left",
        va="center",
        fontsize=CLEAN_LABEL_SIZE,
        color=CLEAN_MEDIUM_GRAY,
    )
    ax.text(
        7.8,
        5.7,
        "Post-AI frontier",
        ha="left",
        va="center",
        fontsize=CLEAN_LABEL_SIZE,
        color=CLEAN_BLACK,
    )
    ax.text(
        14.9,
        POST_FLOOR + 0.25,
        "Slop floor",
        ha="right",
        va="bottom",
        fontsize=CLEAN_SMALL_SIZE,
        color=CLEAN_DARK_GRAY,
    )
    apply_index_frame(ax)
    ax.set_title("AI shifted coding output more than coding quality", loc="left", pad=14)
    ax.set_xlabel("Output volume (schematic index)")
    ax.set_ylabel("Quality (schematic index)")

    fig.text(
        0.08,
        0.03,
        "Schematic values, not measured data. The geometry encodes the argument:\n"
        "the quality ceiling barely moves, volume expands sharply, and many people choose speed over quality.",
        ha="left",
        va="bottom",
        fontsize=CLEAN_SMALL_SIZE,
        color=CLEAN_DARK_GRAY,
    )

    fig.tight_layout(rect=(0.02, 0.1, 0.99, 0.98))
    return fig


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    fig = build_chart()
    for suffix in ("svg", "png"):
        path = output_dir / f"ai_coding_ppf.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
