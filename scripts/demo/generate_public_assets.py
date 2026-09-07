"""Generate public-safe visual assets for the DULoRA demo README.

The figures in this script are conceptual or synthetic. They do not use
private experiment artifacts, private allocation traces, or unpublished method
details.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
MPL_CACHE = ROOT / ".matplotlib-cache"
MPL_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _save(fig: plt.Figure, filename: str) -> None:
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_architecture() -> None:
    labels = [
        "Task data",
        "Utility estimation\n(private in research)",
        "Rank allocation",
        "LoRA rank pattern",
        "Training + evaluation",
    ]
    colors = ["#2b6cb0", "#6b7280", "#2f855a", "#b7791f", "#7c3aed"]

    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    xs = [0.08, 0.29, 0.50, 0.70, 0.90]
    width = 0.16
    height = 0.34
    for index, (x_pos, label, color) in enumerate(zip(xs, labels, colors)):
        rect = FancyBboxPatch(
            (x_pos - width / 2, 0.42),
            width,
            height,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=1.4,
            edgecolor=color,
            facecolor="#f8fafc",
        )
        ax.add_patch(rect)
        ax.text(
            x_pos,
            0.59,
            label,
            ha="center",
            va="center",
            fontsize=10,
            color="#111827",
            weight="bold" if index in {0, 2, 3, 4} else "normal",
        )
        if index < len(xs) - 1:
            arrow = FancyArrowPatch(
                (x_pos + width / 2 + 0.012, 0.59),
                (xs[index + 1] - width / 2 - 0.012, 0.59),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.3,
                color="#475569",
            )
            ax.add_patch(arrow)

    ax.text(
        0.5,
        0.22,
        "Public demo exposes the pipeline shape and software contract; research-specific scoring remains withheld.",
        ha="center",
        va="center",
        fontsize=9,
        color="#475569",
    )
    _save(fig, "public_pipeline.png")


def make_rank_pattern() -> None:
    layers = ["block_a", "block_b", "block_c", "block_d"]
    fixed_rank = [2, 2, 2, 2]
    demo_pattern = [3, 3, 1, 1]

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x_positions = range(len(layers))
    width = 0.34
    ax.bar(
        [x - width / 2 for x in x_positions],
        fixed_rank,
        width,
        label="Fixed rank example",
        color="#64748b",
    )
    ax.bar(
        [x + width / 2 for x in x_positions],
        demo_pattern,
        width,
        label="Illustrative adaptive pattern",
        color="#2f855a",
    )
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(layers)
    ax.set_ylabel("Adapter rank")
    ax.set_ylim(0, 4)
    ax.set_title("Example Non-Uniform Rank Pattern")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        1.5,
        -0.78,
        "Synthetic visualization for public explanation only; not a private DULoRA allocation trace.",
        ha="center",
        va="center",
        fontsize=8.6,
        color="#475569",
        transform=ax.transData,
    )
    _save(fig, "rank_pattern_example.png")


def make_parameter_comparison() -> None:
    labels = ["Fixed LoRA r=8", "Budget-96 setting"]
    relative_params = [100, 50]
    colors = ["#64748b", "#b7791f"]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bars = ax.bar(labels, relative_params, color=colors, width=0.48)
    ax.set_ylabel("Relative trainable adapter parameters")
    ax.set_ylim(0, 115)
    ax.set_title("Reported Adapter Budget Comparison")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, relative_params):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 3,
            f"{value}%",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
        )
    ax.text(
        0.5,
        -19,
        "Safe summary: budget-96 uses 50% fewer trainable adapter parameters than fixed LoRA r=8.",
        ha="center",
        va="center",
        fontsize=8.6,
        color="#475569",
        transform=ax.transData,
    )
    _save(fig, "parameter_comparison.png")


def main() -> None:
    make_architecture()
    make_rank_pattern()
    make_parameter_comparison()


if __name__ == "__main__":
    main()
