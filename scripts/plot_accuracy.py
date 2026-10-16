"""Per-series forecast error chart for the report (docs/images/accuracy_by_series.png).

Dumbbell plot: WAPE of the LightGBM model vs the seasonal-naive baseline for each
store/department series in the most recent 28-day backtest window.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "reports" / "backtest_metrics.json"
OUT = ROOT / "docs" / "images" / "accuracy_by_series.png"

# Validated categorical slots 1-2 (light surface); text uses ink tokens, not series colors.
MODEL, BASELINE = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#1f1f1e", "#5f5e5a", "#e6e5e0"


def main() -> None:
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    rows = [(s["series"], s["model"]["WAPE"], s["seasonal_naive"]["WAPE"]) for s in m["per_series_primary"]]
    rows.sort(key=lambda r: r[2] - r[1])  # biggest improvement ends up at the top
    names = [r[0] for r in rows]
    model = [r[1] for r in rows]
    base = [r[2] for r in rows]
    wins = sum(1 for _, a, b in rows if a < b)
    period = m["primary_test_period"]

    fig, ax = plt.subplots(figsize=(9, 6.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y = range(len(rows))
    ax.hlines(y, [min(a, b) for a, b in zip(model, base)], [max(a, b) for a, b in zip(model, base)],
              color="#c9c8c1", linewidth=2, zorder=1)
    ax.scatter(base, y, s=70, marker="D", color=BASELINE, edgecolor=SURFACE, linewidth=1.5, zorder=3,
               label="Seasonal-naive baseline")
    ax.scatter(model, y, s=80, marker="o", color=MODEL, edgecolor=SURFACE, linewidth=1.5, zorder=4,
               label="LightGBM model")

    ax.set_yticks(list(y), names, fontsize=9.5, color=INK)
    ax.set_xlabel("WAPE (%), lower is better", fontsize=10, color=INK_2)
    ax.tick_params(axis="x", colors=INK_2, labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xlim(0, max(base + model) * 1.12)

    # Direct labels on the top row only; the legend carries identity everywhere else.
    top = len(rows) - 1
    ax.annotate(f"{model[top]:.1f}", (model[top], top), xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=8.5, color=INK)
    ax.annotate(f"{base[top]:.1f}", (base[top], top), xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=8.5, color=INK)

    # Title, subtitle and legend share one left edge at the figure margin.
    fig.text(0.02, 0.975, "Forecast error by store / department", ha="left", va="top",
             fontsize=14, fontweight="bold", color=INK)
    fig.text(0.02, 0.935, f"28-day recursive backtest, {period['from']} to {period['to']}. "
             f"LightGBM beats the baseline on {wins} of {len(rows)} series.",
             ha="left", va="top", fontsize=9.5, color=INK_2)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(0.015, 0.905), ncol=2,
               frameon=False, fontsize=9, labelcolor=INK, handletextpad=0.4, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=SURFACE)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
