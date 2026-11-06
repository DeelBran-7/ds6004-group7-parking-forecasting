"""Build supplementary report figures from published annual summary tables.

Run from any directory: python scripts/build_report_figures.py
No raw parking history or model fitting is required.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#172D3D", "#506478", "#DDE6ED"
BLUE, ORANGE, SLATE = "#006DA3", "#B54B19", "#63768C"
PARKS = ["CSM", "EPM", "HLM", "KAM", "KAS", "PRM", "SLS", "TPM"]


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.tick_params(length=0, colors=MUTED, labelsize=10, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.label.set_color(INK)


def heading(fig, title, subtitle):
    fig.text(.045, .965, title, va="top", fontsize=18, fontweight="bold", color=INK)
    fig.text(.045, .91, subtitle, va="top", fontsize=11, color=MUTED)


def save(fig, path):
    fig.savefig(path, dpi=220, facecolor="white", metadata={"Title": path.stem})
    plt.close(fig)


def quality_figure(frame, path):
    annual = frame.loc[frame["split"].eq("annual")].set_index("carpark_id").loc[PARKS]
    test = frame.loc[frame["split"].eq("test")].set_index("carpark_id").loc[PARKS]
    if len(annual) != 8 or len(test) != 8:
        raise ValueError("Expected eight carparks in each quality split")
    overall_valid = annual["rows_valid_5min"].sum() / annual["rows_total"].sum()
    share = test.loc["KAS", "full_rows_valid"] / test["full_rows_valid"].sum()
    fig, axes = plt.subplots(1, 2, figsize=(12.1, 6.6), gridspec_kw={"width_ratios": [1, 1.3]})
    fig.subplots_adjust(left=.11, right=.96, bottom=.215, top=.76, wspace=.20)
    heading(fig, "Full parking is concentrated in one small carpark",
            f"8 carparks | 365 days | {overall_valid:.2%} valid snapshots | held-out test: 73 days")
    y = np.arange(len(PARKS))
    capacity_ax, risk_ax = axes
    for ax in axes:
        style(ax)
        ax.set(yticks=y, ylim=(7.55, -.55))
    capacity = annual["annual_modal_capacity"]
    capacity_ax.barh(y, capacity, height=.56,
                     color=[ORANGE if p == "KAS" else SLATE for p in PARKS])
    for yi, value in zip(y, capacity):
        capacity_ax.text(value + 12, yi, f"{value:.0f}", color=INK, fontsize=10, va="center")
    capacity_ax.set(xlim=(0, 690), xlabel="Parking spaces (annual modal capacity)", yticks=y)
    capacity_ax.set_yticklabels(PARKS)
    capacity_ax.set_title("Available capacity differs substantially", fontsize=11, loc="left", color=INK, pad=12)
    for label in capacity_ax.get_yticklabels():
        if label.get_text() == "KAS":
            label.set_color(ORANGE)
            label.set_fontweight("bold")
    for sample, offset, color, label in ((annual, -.19, SLATE, "Annual"), (test, .19, BLUE, "Held-out test")):
        values = 100 * sample["full_fraction_valid"]
        risk_ax.barh(y + offset, values, height=.30, color=color, label=label)
        for yi, value in zip(y + offset, values):
            if value == 0:
                risk_ax.scatter([0], [yi], marker="|", s=40, color=color, clip_on=False)
            risk_ax.text(value + .75, yi, f"{value:.2f}%", fontsize=9, color=INK, va="center")
    risk_ax.set(xlim=(0, 54), xlabel="Full snapshots (% of valid observations)", yticklabels=[])
    risk_ax.set_title("Full means zero reported free spaces", fontsize=11, loc="left", color=INK, pad=12)
    risk_ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.19), ncols=2, frameon=False,
                   fontsize=10, labelcolor=INK)
    fig.text(.045, .125, f"KAS contributes {share:.1%} of full test snapshots. KAM, the closest carpark, has none.",
             fontsize=11, color=INK, fontweight="bold")
    fig.text(.045, .047,
             "Rates describe carpark-time observations, not individual drivers. Denominators exclude invalid readings.\n"
             "PRM capacity is usually 576; three training records report 234 and remain flagged for source verification.",
             fontsize=9.3, color=MUTED, linespacing=1.5)
    save(fig, path)


def carpark_forecast_figure(frame, path):
    data = frame.loc[frame["cohort"].eq("all") & frame["model"].isin(["B0", "M0", "M1"])]
    if data.duplicated(["carpark_id", "horizon_min", "model"]).any():
        raise ValueError("Expected one model score per carpark and horizon")
    model_specs = [("B0", -.23, SLATE, "B0 Current occupancy"),
                   ("M0", 0, BLUE, "M0 XGBoost"),
                   ("M1", .23, ORANGE, "M1 XGBoost + holidays")]
    fig, axes = plt.subplots(1, 2, figsize=(12.1, 7.5))
    fig.subplots_adjust(left=.10, right=.97, bottom=.19, top=.77, wspace=.18)
    heading(fig, "Forecast accuracy varies across carparks",
            "Held-out MAE on shared B0/B1/M0/M1 observations within each carpark; lower is better")
    fig.legend(handles=[Patch(facecolor=color, label=label) for _, _, color, label in model_specs],
               loc="upper center", bbox_to_anchor=(.52, .864), ncols=3, frameon=False,
               fontsize=10, labelcolor=INK)
    y = np.arange(len(PARKS))
    for horizon, ax in zip((15, 30), axes):
        style(ax)
        sample = data.loc[data["horizon_min"].eq(horizon)]
        coverage = sample.groupby("carpark_id")["n_scored"].nunique()
        if not coverage.eq(1).all():
            raise ValueError("Each carpark must use shared model samples")
        for model, offset, color, _ in model_specs:
            rows = sample.loc[sample["model"].eq(model)].set_index("carpark_id").loc[PARKS]
            values = rows["mae"] * 100
            ax.barh(y + offset, values, height=.19, color=color)
            for yi, value in zip(y + offset, values):
                ax.text(value + .06, yi, f"{value:.2f}", va="center", fontsize=8.8, color=INK)
        ax.set(yticks=y, yticklabels=PARKS if horizon == 15 else [], ylim=(7.6, -.6),
               xlim=(0, 5), xlabel="MAE (percentage points)", xticks=np.arange(6))
        ax.set_title(f"{horizon} minutes", loc="left", fontsize=12, fontweight="bold", color=INK, pad=12)
        if horizon == 15:
            for label in ax.get_yticklabels():
                if label.get_text() == "KAS":
                    label.set_fontweight("bold")
                    label.set_color(ORANGE)
    kas = data.loc[data["carpark_id"].eq("KAS") & data["horizon_min"].eq(15)].set_index("model")["mae"] * 100
    fig.text(.045, .10, f"KAS at 15 minutes: B0 {kas['B0']:.3f} pp, M0 {kas['M0']:.3f} pp, M1 {kas['M1']:.3f} pp.",
             fontsize=11, fontweight="bold", color=INK)
    fig.text(.045, .045,
             "Overall improvement is not universal. Bars are point estimates; no per-carpark significance claim is made.\n"
             "The held-out test contains no public holidays, so this comparison cannot establish a holiday benefit.",
             fontsize=9.3, linespacing=1.5, color=MUTED)
    save(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-root", type=Path, default=ROOT / "examples" / "annual")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "annual" / "figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    quality_figure(pd.read_csv(args.summary_root / "data_quality" / "quality_by_carpark.csv"),
                   args.output / "data_quality.png")
    carpark_forecast_figure(pd.read_csv(args.summary_root / "evaluation" / "forecast_metrics_by_carpark.csv"),
                           args.output / "forecast_by_carpark.png")
    print(f"Wrote data_quality.png and forecast_by_carpark.png to {args.output}")


if __name__ == "__main__":
    main()
