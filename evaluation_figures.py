"""Render static figures from Part C's held-out evaluation tables."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


HORIZONS = (15, 30)
INK, MUTED, GRID = "#172D3D", "#506478", "#DDE6ED"
BLUE, ORANGE, SLATE = "#006DA3", "#B54B19", "#63768C"
MODEL_LABELS = {"B0": "B0  Current occupancy", "B1": "B1  Previous week",
                "M0": "M0  XGBoost", "M1": "M1  XGBoost + holidays"}
STRATEGY_LABELS = {"A": "A  Nearest", "B": "B  Most free now", "C0": "C0  Current occupancy",
                   "C": "C  Forecast occupancy", "C_XGB_RISK": "XGBoost  Full risk",
                   "C_CAT_RISK": "CatBoost  Full risk"}


def _heading(fig, title, subtitle, *, y=0.965):
    fig.text(0.045, y, title, va="top", fontsize=16, color=INK, fontweight="bold")
    fig.text(0.045, y - 0.06, subtitle, va="top", fontsize=9.5, color=MUTED)


def _footer(fig, text):
    fig.text(0.045, 0.035, text, va="bottom", fontsize=8.3, linespacing=1.5, color=MUTED)


def _style(ax, axis="x"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.tick_params(length=0, colors=MUTED, labelsize=9, pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _save(fig, path):
    fig.savefig(path, dpi=180, facecolor="white", metadata={"Title": path.stem.replace("_", " ")})
    plt.close(fig)
    return path


def _count_text(values, suffix):
    counts = sorted(pd.Series(values).dropna().astype(int).unique())
    if not counts:
        return f"no {suffix}"
    if len(counts) == 1:
        return f"{counts[0]:,} {suffix}"
    return f"{counts[0]:,}–{counts[-1]:,} {suffix}"


def _mae_plot(metrics, path):
    metrics = metrics.loc[metrics["cohort"].eq("all") & metrics["model"].isin(MODEL_LABELS)]
    labels = list(MODEL_LABELS)
    positive = metrics["mae"].dropna() * 100
    upper = max(1.0, float(positive.max())) if not positive.empty else 1.0
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.45))
    fig.subplots_adjust(left=0.245, right=0.97, top=0.76, bottom=0.225, wspace=0.32)
    _heading(fig, "Occupancy forecast error", "Held-out tests · mean absolute error · lower is better")
    for horizon, ax in zip(HORIZONS, axes):
        rows = metrics.loc[metrics["horizon_min"].eq(horizon)].set_index("model").reindex(labels)
        ax.set_title(f"{horizon} minutes  |  {_count_text(rows['n_scored'], 'scored rows')}", loc="left",
                     fontsize=10, color=INK, fontweight="bold", pad=12)
        _style(ax)
        values = rows["mae"] * 100
        for index, (name, value) in enumerate(values.items()):
            if pd.notna(value):
                ax.barh(index, value, height=0.54, color=SLATE if name.startswith("B") else BLUE)
                ax.text(value + upper * 0.028, index, f"{value:.2f}", va="center", color=INK, fontsize=9)
            else:
                ax.text(upper * 0.028, index, "Not scored", va="center", color=MUTED, fontsize=8.5)
        ax.set(xlim=(0, upper * 1.22), yticks=range(len(labels)), ylim=(len(labels) - 0.5, -0.5),
               xlabel="MAE (percentage points)")
        ax.set_yticklabels([MODEL_LABELS[name] for name in labels] if horizon == HORIZONS[0] else [])
    _footer(fig, "One observation is a carpark and departure time. B0/B1 are baselines; M0/M1 are fitted models.")
    return _save(fig, path)


def _calibration_plot(reliability, path):
    rows = reliability.loc[reliability["event"].eq("full") & reliability["cohort"].eq("open_at_departure")
                           & reliability["n"].gt(0) & reliability["model"].isin(["xgboost", "catboost"])]
    if "version" in rows:
        rows = rows.loc[rows["version"].eq("final")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.45))
    fig.subplots_adjust(left=0.11, right=0.96, top=0.755, bottom=0.235, wspace=0.29)
    _heading(fig, "Calibration of full-at-arrival risk", "Held-out tests · carparks with at least one free space at departure")
    specifications = {"xgboost": (BLUE, "o", "XGBoost"), "catboost": (ORANGE, "s", "CatBoost")}
    for horizon, ax in zip(HORIZONS, axes):
        _style(ax, "both")
        ax.set_title(f"{horizon} minutes", loc="left", fontsize=11, color=INK, fontweight="bold", pad=10)
        ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1, alpha=0.7, zorder=1)
        block = rows.loc[rows["horizon_min"].eq(horizon)].dropna(subset=["mean_predicted", "event_rate"])
        for name, (color, marker, _) in specifications.items():
            for is_sparse in (False, True):
                sample = block.loc[block["model"].eq(name) & block["n"].lt(30).eq(is_sparse)]
                ax.scatter(sample["mean_predicted"], sample["event_rate"],
                           s=24 + 17 * np.log10(sample["n"] + 1), marker=marker,
                           facecolors="white" if is_sparse else color, edgecolors=color,
                           alpha=0.82, linewidth=1.25, zorder=3)
        if block.empty:
            ax.text(0.5, 0.2, "No scored bins", color=MUTED, ha="center", transform=ax.transAxes)
        ax.set(xlim=(-0.035, 1.035), ylim=(-0.035, 1.035), xlabel="Mean predicted probability",
               xticks=np.linspace(0, 1, 5), yticks=np.linspace(0, 1, 5))
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[0].set_ylabel("Observed full-at-arrival rate")
    handles = [Line2D([0], [0], marker=marker, color=color, linestyle="none", markersize=6, label=label)
               for color, marker, label in specifications.values()]
    handles.append(Line2D([0], [0], color=MUTED, linestyle="--", linewidth=1, label="Perfect calibration"))
    fig.legend(handles=handles, ncols=3, loc="upper center", bbox_to_anchor=(0.52, 0.873),
               fontsize=9, frameon=False, labelcolor=INK)
    _footer(fig, "Each point is a nonempty 10-point probability bin. Markers grow with bin count; hollow markers have fewer than 30 rows.\n"
                 "Full at arrival means zero free spaces. Points are observed rates; no confidence intervals are shown.")
    return _save(fig, path)


def _tradeoff_plot(metrics, path):
    labels = list(STRATEGY_LABELS)
    rows = metrics.loc[metrics["cohort"].eq("visitor_eligible") & metrics["strategy"].isin(labels)]
    if "distance_weight" in rows and rows.duplicated(["horizon_min", "strategy"]).any():
        raise ValueError("Tradeoff figure requires one validation-selected weight per strategy and horizon")
    fig, axes = plt.subplots(2, 2, figsize=(11.05, 7.35))
    fig.subplots_adjust(left=0.245, right=0.968, top=0.815, bottom=0.155, wspace=0.22, hspace=0.58)
    _heading(fig, "Distance and near-full arrivals", "First-choice outcomes on the same held-out departures")
    fields = ("mean_extra_vs_nearest_m", "top1_near_full_rate")
    upper = [max(1.0, float(rows[fields[0]].max())), max(0.001, float(rows[fields[1]].max()))]
    captions = ("Extra distance vs nearest (m)", "Arrivals with <10% free spaces")
    formats = (lambda x: f"{x:.2f}", lambda x: f"{100*x:.2f}%")
    for horizon, ax_row in zip(HORIZONS, axes):
        data = rows.loc[rows["horizon_min"].eq(horizon)].set_index("strategy").reindex(labels)
        for col, ax in enumerate(ax_row):
            title = f"{horizon} min  |  {_count_text(data['matched_cases'], 'matched cases')}" if col == 0 else f"{horizon} min  |  Near-full rate"
            ax.set_title(title, loc="left", color=INK, fontweight="bold", fontsize=10, pad=8)
            _style(ax)
            ax.axvline(0, color=GRID, linewidth=1)
            for index, (name, value) in enumerate(data[fields[col]].items()):
                if pd.isna(value):
                    continue
                color = ORANGE if name.endswith("RISK") else BLUE if name == "C" else SLATE
                marker = "s" if name.endswith("RISK") else "D" if name == "C" else "o"
                ax.hlines(index, 0, value, color=color, alpha=0.35, linewidth=1.8)
                ax.scatter(value, index, color=color, marker=marker, s=34, zorder=3)
                ax.text(value + upper[col] * 0.05, index, formats[col](value), va="center", color=INK, fontsize=8.5)
            ax.set(xlim=(-upper[col] * 0.025, upper[col] * 1.25), yticks=range(len(labels)),
                   ylim=(len(labels) - 0.5, -0.5), xlabel=captions[col])
            ax.set_yticklabels([STRATEGY_LABELS[name] for name in labels] if col == 0 else [])
            if col == 1:
                ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    full_counts = rows["top1_full_count"]
    full_note = ("No first-choice carpark was fully occupied on arrival in this comparison."
                 if not full_counts.empty and full_counts.notna().all() and full_counts.eq(0).all()
                 else "Full-at-arrival counts are reported separately in the results table.")
    _footer(fig, f"{full_note}\nNear-full is a capacity percentage. Distance is straight-line distance, not walking distance.")
    return _save(fig, path)


def render_evaluation(output_dir):
    """Render three PNGs alongside the corresponding Part C tables."""
    output = Path(output_dir)
    return [
        _mae_plot(pd.read_csv(output / "forecast_metrics.csv"), output / "forecast_mae.png"),
        _calibration_plot(pd.read_csv(output / "risk_reliability.csv"), output / "full_risk_calibration.png"),
        _tradeoff_plot(pd.read_csv(output / "recommendation_metrics.csv"), output / "recommendation_tradeoff.png"),
    ]
