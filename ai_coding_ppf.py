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

X_MAX = 11.5
Y_MAX = 10.0
POST_CEILING = 9.45
POST_CLIFF_LOW = 3.2
POST_CENTER = 4.2
POST_SCALE = 0.65
TAIL_START = 6.0
TAIL_DECAY = 0.16
TAIL_EXPONENT = 1.1


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
    x = 0.25 + 1.55 * t
    y = 9.2 - 6.8 * (t**1.85)
    return x, y


def post_ai_quality(x: float | np.ndarray) -> float | np.ndarray:
    """Return the post-AI frontier quality at a given output index."""
    x_array = np.asarray(x)
    cliff_quality = POST_CLIFF_LOW + (POST_CEILING - POST_CLIFF_LOW) / (
        1 + np.exp((x_array - POST_CENTER) / POST_SCALE)
    )
    tail_drag = TAIL_DECAY * np.maximum(x_array - TAIL_START, 0) ** TAIL_EXPONENT
    quality = np.maximum(cliff_quality - tail_drag, 0)
    if np.isscalar(x):
        return float(quality)
    return quality


def post_ai_volume_at_quality(quality: float) -> float:
    """Invert the post-AI frontier before the high-volume deterioration tail."""
    ratio = (POST_CEILING - POST_CLIFF_LOW) / (quality - POST_CLIFF_LOW) - 1
    return POST_CENTER + POST_SCALE * np.log(ratio)


def apply_index_frame(ax: plt.Axes) -> None:
    """Apply bounded axes for the schematic index ranges."""
    ax.set_xlim(0, X_MAX + 0.4)
    ax.set_ylim(0, Y_MAX + 0.25)
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["left"].set_position(("data", 0))
    ax.spines["bottom"].set_bounds(0, X_MAX)
    ax.spines["left"].set_bounds(0, Y_MAX)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(colors=CLEAN_BLACK)


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
    post_x = np.linspace(0.25, X_MAX, 520)
    post_y = post_ai_quality(post_x)

    baseline_x = 1.0
    baseline_y = float(np.interp(baseline_x, pre_x, pre_y))
    baseline = (baseline_x, baseline_y)

    careful = (baseline_x, post_ai_quality(baseline_x))
    disciplined = (post_ai_volume_at_quality(baseline_y), baseline_y)
    vibe = (10.0, post_ai_quality(10.0))

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
        (1.25, 10.2),
        va="top",
    )
    draw_move(
        ax,
        baseline,
        disciplined,
        TOL_INDIGO,
        "B. Disciplined volume\nmore volume, same quality",
        (3.95, 7.45),
        va="bottom",
    )
    draw_move(
        ax,
        baseline,
        vibe,
        TOL_ROSE,
        "C. Vibe coding",
        (10.2, 3.0),
        lw=1.8,
    )

    ax.text(
        baseline_x - 0.08,
        baseline_y - 0.45,
        "Pre-AI\nbaseline",
        ha="right",
        va="top",
        fontsize=CLEAN_SMALL_SIZE,
        color=CLEAN_BLACK,
        linespacing=1.1,
    )
    ax.text(
        1.9,
        2.6,
        "Pre-AI frontier",
        ha="left",
        va="center",
        fontsize=CLEAN_LABEL_SIZE,
        color=CLEAN_MEDIUM_GRAY,
    )
    ax.text(
        5.75,
        2.65,
        "Post-AI frontier",
        ha="left",
        va="center",
        fontsize=CLEAN_LABEL_SIZE,
        color=CLEAN_BLACK,
    )
    apply_index_frame(ax)
    ax.set_xlabel("Output volume")
    ax.set_ylabel("Quality")

    fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.99))
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
