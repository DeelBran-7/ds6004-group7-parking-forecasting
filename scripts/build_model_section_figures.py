"""Render concise model-report figures from published evaluation summaries only.

No fitting, selection, or modification of source results is performed.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/history_comparison/two_year_standard/evaluation"
OUT = ROOT / "reports/model_section/figures"
INK = "#172B3A"
MUTED = "#526776"
COLORS = {"B0": "#8B9DA8", "B1": "#B9C4CB", "M0": "#087D84", "M1": "#C57935"}


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#BFC9CF")
    ax.tick_params(length=3, color="#A5B2BB", labelcolor=INK)
    ax.set_axisbelow(True)


def forecast_performance():
    overall = pd.read_csv(SOURCE / "forecast_metrics.csv")
    overall = overall.loc[overall.cohort.eq("all")]
    park = pd.read_csv(SOURCE / "forecast_metrics_by_carpark.csv")
    park = park.loc[park.cohort.eq("all") & park.horizon_min.eq(30)
                    & park.model.isin(["B0", "M0"])]
    models = ["B0", "B1", "M0", "M1"]
    if len(overall) != 8 or overall.duplicated(["horizon_min", "model"]).any():
        raise ValueError("Expected four models at both horizons")
    if park.carpark_id.nunique() != 8 or len(park) != 16:
        raise ValueError("All eight car parks must appear in the comparison")
    for _, rows in overall.groupby("horizon_min"):
        if rows.n_scored.nunique() != 1:
            raise ValueError("Overall scores do not share a sample")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.1, 3.45), gridspec_kw={"width_ratios": [1.07, 1]})
    fig.subplots_adjust(left=.075, right=.985, top=.85, bottom=.28, wspace=.32)
    width = .18
    for i, model in enumerate(models):
        rows = overall.loc[overall.model.eq(model)].set_index("horizon_min").loc[[15, 30]]
        x = np.arange(2) + (i - 1.5) * width
        values = rows.mae.to_numpy() * 100
        bars = ax.bar(x, values, width, color=COLORS[model], label=model)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=3, fontsize=8)
    ax.set_xticks([0, 1], ["15 minutes", "30 minutes"])
    ax.set_ylim(0, 5.75)
    ax.set_ylabel("MAE in percentage points", fontsize=9)
    ax.set_title("a  Overall forecast error", loc="left", fontsize=10, fontweight="bold", color=INK, pad=12)
    ax.grid(axis="y", color="#E8EDF0", linewidth=.6)
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(-.02, -.14), frameon=False,
              columnspacing=1.15, handlelength=.9, handletextpad=.4, fontsize=8)
    clean(ax)

    data = park.pivot(index="carpark_id", columns="model", values="mae").sort_index() * 100
    y = np.arange(len(data))
    bx.hlines(y, data.M0, data.B0, color="#CCD7DE", linewidth=2)
    bx.scatter(data.B0, y, s=28, color=COLORS["B0"], label="B0", zorder=3)
    bx.scatter(data.M0, y, s=28, color=COLORS["M0"], label="M0", zorder=3)
    bx.set_yticks(y, data.index)
    bx.invert_yaxis()
    bx.set_xlim(0, 5.75)
    bx.set_xlabel("30-minute MAE in percentage points", fontsize=9, labelpad=7)
    bx.set_title("b  Every study car park", loc="left", fontsize=10, fontweight="bold", color=INK, pad=12)
    bx.grid(axis="x", color="#E8EDF0", linewidth=.6)
    clean(bx)
    counts = overall.drop_duplicates("horizon_min").set_index("horizon_min").n_scored
    fig.text(.075, .055, f"Matched test samples: {counts[15]:,} / {counts[30]:,} at 15 / 30 minutes. Lower MAE is better.",
             fontsize=8.2, color=MUTED)
    fig.savefig(OUT / "forecast_performance.png", dpi=300, facecolor="white")
    plt.close(fig)


def arrival_risk_calibration():
    bins = pd.read_csv(SOURCE / "risk_reliability.csv")
    metrics = pd.read_csv(SOURCE / "risk_probability_metrics.csv")
    mask = lambda d: d.horizon_min.eq(30) & d.event.eq("full") & d.cohort.eq("open_at_departure")
    bins = bins.loc[mask(bins)]
    metrics = metrics.loc[mask(metrics)].set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.75))
    fig.subplots_adjust(left=.085, right=.985, top=.83, bottom=.23, wspace=.23)
    for ax, model, color, title in zip(axes, ["xgboost", "catboost"], ["#087D84", "#C57935"], ["XGBoost", "CatBoost"]):
        rows = bins.loc[bins.model.eq(model) & bins.n.gt(0)].copy()
        stat = metrics.loc[model]
        if int(rows.n.sum()) != int(stat.n_scored) or int(rows.events.sum()) != int(stat.events):
            raise ValueError("Calibration bins do not reconstruct probability metrics")
        ax.plot([0, 1], [0, 1], color="#9CAAB4", linewidth=1, linestyle="--", zorder=1)
        for row in rows.itertuples():
            ax.scatter([row.mean_predicted], [row.event_rate], s=30, edgecolors=color,
                       facecolors="white" if row.n < 30 else color, linewidths=1.2, zorder=3)
            # Offset the last two XGBoost labels on opposite sides to avoid collision.
            offset = (4, -13) if model == "xgboost" and .8 < row.mean_predicted < .9 else (4, 6)
            if row.mean_predicted > .88:
                offset = (5, -12) if model == "catboost" else (5, 3)
            ax.annotate(f"{int(row.n):,}", (row.mean_predicted, row.event_rate),
                        xytext=offset, textcoords="offset points", fontsize=7.3, color=INK,
                        ha="left")
        ax.set_xlim(-.035, 1.07); ax.set_ylim(-.035, 1.07)
        ax.set_xticks(np.linspace(0, 1, 6)); ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_xlabel("Predicted full-arrival probability", fontsize=8.7)
        ax.set_title(f"{title}  |  Brier {stat.brier:.6f}", loc="left", fontsize=9.8,
                     color=INK, fontweight="bold", pad=10)
        ax.grid(color="#E8EDF0", linewidth=.6)
        clean(ax)
    axes[0].set_ylabel("Observed full-arrival rate", fontsize=9)
    stat = metrics.loc["xgboost"]
    fig.text(.085, .94, f"30 minutes  |  Spaces available at departure  |  {int(stat.n_scored):,} samples, {int(stat.events):,} full arrivals",
             fontsize=9, color=INK)
    fig.text(.085, .10, "Fixed 0.1-wide probability bins. Labels show bin counts; hollow markers denote fewer than 30 samples.",
             fontsize=8, color=MUTED)
    fig.text(.085, .045, "Points are estimates without uncertainty intervals; sparse high-probability bins do not establish calibration.",
             fontsize=8, color=MUTED)
    fig.savefig(OUT / "arrival_risk_calibration.png", dpi=300, facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
    forecast_performance()
    arrival_risk_calibration()
    print(OUT)


if __name__ == "__main__":
    main()
