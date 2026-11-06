"""Describe fixed departure-time subgroups of a completed parking study.

This is a retrospective supplement, not a new held-out confirmation. It does
not fit models, select ranking weights or redefine the primary arrival event.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate import (KEY, RANKING_FIELDS, correct_family, evaluate_recommendations,
                      read_inputs, require_same_values)
from evaluation_stats import paired_day_inference


MODELS = ("B0", "B1", "M0", "M1")
STRATEGIES = ("A", "B", "C0", "C", "C_XGB_RISK", "C_CAT_RISK")
GROUPS = ("overall", "evening_peak", "weekend")
GROUP_LABELS = {"overall": "All departures", "evening_peak": "Evening: 17:00-20:00",
                "weekend": "Weekend departures"}


def subgroup_masks(decision_times):
    """Pre-fixed, overlapping masks based only on local departure information."""
    times = pd.to_datetime(decision_times)
    if times.isna().any() or times.dt.tz is not None:
        raise ValueError("Use nonmissing, naive Singapore departure timestamps")
    return {"overall": pd.Series(True, index=times.index),
            "evening_peak": times.dt.hour.ge(17) & times.dt.hour.lt(20),
            "weekend": times.dt.dayofweek.ge(5)}


def forecast_tables(test, *, repetitions, seed):
    eligible = (test["current_is_valid"].eq(True) & test["actual_is_observed"].eq(True)
                & test["forecast_label_eligible"].eq(True))
    free, capacity = test.loc[eligible, "actual_lots_available"], test.loc[eligible, "actual_total_lots"]
    if not (np.isfinite(capacity) & capacity.gt(0) & np.isfinite(free) & free.between(0, capacity)).all():
        raise ValueError("Eligible forecasts need valid arrival counts")
    observed = 1 - test["actual_lots_available"] / test["actual_total_lots"]
    require_same_values(test.loc[eligible, "actual_occupancy"], observed.loc[eligible],
                        "Occupancy labels disagree with observed arrival counts")
    predictions = test[list(MODELS)]
    finite = np.isfinite(predictions)
    if (finite & ~(predictions.ge(0) & predictions.le(1))).any().any():
        raise ValueError("Finite occupancy predictions must lie in [0, 1]")
    common = eligible & finite.all(axis=1)
    summaries, pairs = [], []
    for subgroup, group_mask in subgroup_masks(test["decision_time"]).items():
        for horizon in (15, 30):
            relevant = group_mask & test["horizon_min"].eq(horizon)
            keep = relevant & common
            rows = test.loc[keep]
            days = rows["decision_time"].dt.normalize()
            losses = {}
            for model in MODELS:
                errors = rows[model] - observed.loc[keep]
                losses[model] = errors.abs()
                summaries.append({"subgroup": subgroup, "horizon_min": horizon, "model": model,
                                  "n_eligible": int((relevant & eligible).sum()), "n_scored": len(rows),
                                  "decision_dates": days.nunique(), "mae": errors.abs().mean(),
                                  "rmse": np.sqrt(errors.pow(2).mean()),
                                  "actual_full_count": int(rows["actual_lots_available"].eq(0).sum()),
                                  "actual_near_full_count": int(rows["actual_lots_available"].mul(10)
                                                                  .lt(rows["actual_total_lots"]).sum())})
            for candidate, reference in (("M0", "B0"), ("M1", "M0")):
                stats = paired_day_inference(days, losses[candidate], losses[reference],
                                             repetitions=repetitions, seed=seed)
                reference_mean, candidate_mean = stats["reference_mean"], stats["candidate_mean"]
                relative = ((reference_mean - candidate_mean) / reference_mean
                            if reference_mean is not None and reference_mean > 0 else None)
                pairs.append({"subgroup": subgroup, "horizon_min": horizon,
                              "comparison": f"{candidate}-{reference}", "metric": "mae",
                              "relative_reduction": relative, **stats})
    return pd.DataFrame(summaries), correct_family(pd.DataFrame(pairs))


def validate_rankings(rankings, test, settings):
    if set(rankings["strategy"]) != set(STRATEGIES):
        raise ValueError("All six original strategies are required")
    if set(rankings["cohort"]) != {"visitor_eligible", "open_at_departure"}:
        raise ValueError("Both original candidate cohorts are required")
    if not rankings["distance_weight"].eq(settings["selected_distance_weight"]).all():
        raise ValueError("Ranking weights differ from the original validation selection")
    fields = [*KEY, "target_time", "current_lots_available", "current_is_valid", "actual_is_observed",
              "actual_lots_available", "actual_total_lots", settings["forecast_column"]]
    checked = rankings.merge(test[fields], on=KEY, how="left", validate="many_to_one",
                             suffixes=("", "_source"), indicator=True)
    if (not checked["_merge"].eq("both").all() or not checked["current_is_valid"].eq(True).all()
            or not checked["actual_is_observed"].eq(True).all()):
        raise ValueError("Every ranked outcome must be a valid held-out endpoint")
    for column in ("target_time", "current_lots_available", "actual_lots_available", "actual_total_lots"):
        require_same_values(checked[column], checked[f"{column}_source"], f"Ranked {column} disagrees")
    require_same_values(checked["forecast_occupancy"], checked[settings["forecast_column"]],
                        "Rankings use a different occupancy forecast")
    free, capacity = checked["actual_lots_available"], checked["actual_total_lots"]
    if not (np.isfinite(free) & np.isfinite(capacity) & capacity.gt(0) & free.between(0, capacity)).all():
        raise ValueError("Invalid ranked arrival counts")
    labels = {"full_at_arrival": free.eq(0), "near_full_at_arrival": free.mul(10).lt(capacity),
              "low_spaces_at_arrival": free.lt(settings["low_spaces_threshold"])}
    for column, observed in labels.items():
        require_same_values(checked[column], observed, f"Ranked {column} definition disagrees")
    if checked.loc[checked["cohort"].eq("open_at_departure"), "current_lots_available"].le(0).any():
        raise ValueError("Open-at-departure candidates must have at least one free space")


def policy_tables(rankings, *, repetitions, seed):
    summaries, pairs = [], []
    for subgroup, mask in subgroup_masks(rankings["decision_time"]).items():
        metrics, comparisons = evaluate_recommendations(rankings.loc[mask], repetitions=repetitions, seed=seed)
        summaries.append(metrics.assign(subgroup=subgroup))
        # The original primary event stays full-at-arrival. Subgroup analyses are exploratory.
        comparisons["primary_in_original_protocol"] = comparisons.pop("primary")
        comparisons["primary"] = (comparisons["primary_in_original_protocol"] & (subgroup == "overall"))
        comparisons["changed_first_choice_rate"] = comparisons["changed_first_choices"] / comparisons["n_pairs"]
        pairs.append(comparisons.assign(subgroup=subgroup))
    return pd.concat(summaries, ignore_index=True), correct_family(pd.concat(pairs, ignore_index=True))


def check_overall_against_published(forecasts, policies, evaluation_dir):
    """The added subgroup code must preserve the original full-test estimates."""
    published = pd.read_csv(evaluation_dir / "forecast_metrics.csv")
    published = published.loc[published["cohort"].eq("all")].set_index(["horizon_min", "model"]).sort_index()
    current = forecasts.loc[forecasts["subgroup"].eq("overall")].set_index(["horizon_min", "model"]).sort_index()
    if not current.index.equals(published.index):
        raise ValueError("Overall forecast keys disagree with the published study")
    for column in ("n_eligible", "n_scored", "mae", "rmse"):
        require_same_values(current[column], published[column], f"Overall forecast {column} changed")
    published = pd.read_csv(evaluation_dir / "recommendation_metrics.csv")
    keys = ["horizon_min", "cohort", "strategy"]
    published = published.set_index(keys).sort_index()
    current = policies.loc[policies["subgroup"].eq("overall")].set_index(keys).sort_index()
    if not current.index.equals(published.index):
        raise ValueError("Overall policy keys disagree with the published study")
    for column in published.columns:
        require_same_values(current[column], published[column], f"Overall policy {column} changed")


def make_figures(forecasts, policies, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"B0": "#63768C", "B1": "#C7D4DF", "M0": "#006DA3", "M1": "#C36B26"}
    fig, axes = plt.subplots(1, 2, figsize=(11.1, 5.5), sharey=True)
    fig.subplots_adjust(left=.08, right=.98, top=.77, bottom=.23, wspace=.12)
    fig.text(.06, .96, "Occupancy error by departure-time subgroup", fontsize=17, fontweight="bold", color="#172D3D")
    fig.text(.06, .902, "Same fitted models and original held-out dates | lower MAE is better", fontsize=11, color="#506478")
    for horizon, ax in zip((15, 30), axes):
        x = np.arange(len(GROUPS))
        for i, model in enumerate(MODELS):
            rows = forecasts.loc[forecasts["horizon_min"].eq(horizon) & forecasts["model"].eq(model)]
            values = rows.set_index("subgroup").loc[list(GROUPS), "mae"] * 100
            ax.bar(x + (i - 1.5) * .18, values, width=.17, color=colors[model], label=model)
        ax.set_xticks(x, ["Overall", "17:00-20:00", "Weekend"])
        ax.set_title(f"{horizon} minutes", fontsize=12, fontweight="bold", loc="left")
        ax.grid(axis="y", color="#DDE6ED", linewidth=.7)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0, colors="#506478")
    axes[0].set_ylabel("MAE (percentage points)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, .865), ncols=4, frameon=False)
    fig.text(.06, .11, "Subgroups use Singapore departure time; evening and weekend can overlap.", fontsize=9, color="#506478")
    fig.text(.06, .065, "Within-subgroup paired tests are in CSV. This figure does not test differences in gains between subgroups.", fontsize=9, color="#506478")
    fig.savefig(output / "forecast_subgroups.png", dpi=220, facecolor="white")
    plt.close(fig)

    labels = ["A Nearest", "B Most spaces now", "C0 Current occupancy", "C Forecast occupancy", "XGB Full risk", "Cat Full risk"]
    policy_colors = ["#63768C", "#A7B8C7", "#006DA3", "#087D84", "#C36B26", "#975022"]
    fig, axes = plt.subplots(3, 2, figsize=(11.1, 9.6))
    fig.subplots_adjust(left=.23, right=.97, top=.87, bottom=.12, wspace=.28, hspace=.58)
    fig.text(.045, .965, "Arrival outcomes by departure-time subgroup", fontsize=17, fontweight="bold", color="#172D3D")
    fig.text(.045, .924, "30-minute replay | all candidate carparks have spaces at departure", fontsize=10.8, color="#506478")
    selected = policies.loc[policies["horizon_min"].eq(30) & policies["cohort"].eq("open_at_departure")]
    for row_i, subgroup in enumerate(GROUPS):
        rows = selected.loc[selected["subgroup"].eq(subgroup)].set_index("strategy").loc[list(STRATEGIES)]
        for col_i, (rate, count, title) in enumerate((("top1_full_rate", "top1_full_count", "Full: 0 spaces"),
                                                     ("top1_near_full_rate", "top1_near_full_count", "Near-full: <10% free"))):
            ax = axes[row_i, col_i]
            values = rows[rate] * 100
            y = np.arange(len(STRATEGIES))
            ax.barh(y, values, color=policy_colors, height=.62)
            upper = max(.02, float(selected[rate].max() * 100))
            for pos, value, events in zip(y, values, rows[count]):
                ax.text(value + upper * .025, pos, f"{value:.3f}% ({int(events)})", va="center", fontsize=8, color="#172D3D")
            ax.set(yticks=y, yticklabels=labels if col_i == 0 else [], ylim=(5.65, -.65), xlim=(0, upper * 1.65))
            ax.set_title(f"{GROUP_LABELS[subgroup]} | n={int(rows['matched_cases'].iloc[0]):,}\n{title}",
                         loc="left", fontsize=10, fontweight="bold", color="#172D3D", pad=8)
            ax.grid(axis="x", color="#DDE6ED", linewidth=.7)
            ax.set_axisbelow(True)
            ax.tick_params(length=0, colors="#506478", labelsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.text(.045, .057, "Labels show rates and event counts. Full remains the primary event; near-full includes full and is a separate outcome.", fontsize=9, color="#506478")
    fig.text(.045, .025, "Distance and both arrival horizons are reported in CSV. Rare or zero events do not establish policy equivalence.", fontsize=9, color="#506478")
    fig.savefig(output / "recommendation_subgroups.png", dpi=220, facecolor="white")
    plt.close(fig)


def run(study_dir, output_dir, *, repetitions=10_000, seed=20_261_105, figures=True):
    study, output = Path(study_dir), Path(output_dir)
    forecasts = read_inputs(study / "forecast/predictions.csv")
    if forecasts.empty or forecasts[KEY].isna().any().any() or forecasts.duplicated(KEY).any():
        raise ValueError("Forecasts must have unique nonmissing keys")
    if set(forecasts["split"]) != {"validation", "test"}:
        raise ValueError("Expected the existing validation and test predictions")
    test = forecasts.loc[forecasts["split"].eq("test")].copy()
    if forecasts.loc[forecasts["split"].eq("validation"), "target_time"].max() >= test["decision_time"].min():
        raise ValueError("Validation targets overlap the test period")
    settings = json.loads((study / "recommendation/replay_settings.json").read_text())
    if settings["low_spaces_threshold"] != 3:
        raise ValueError("This supplement pre-specifies the existing threshold of 3 spaces")
    rankings = read_inputs(study / "recommendation/recommendation_rankings.csv", usecols=set(RANKING_FIELDS).__contains__)
    validate_rankings(rankings, test, settings)
    forecast_metrics, forecast_pairs = forecast_tables(test, repetitions=repetitions, seed=seed)
    policy_metrics, policy_pairs = policy_tables(rankings, repetitions=repetitions, seed=seed)
    check_overall_against_published(forecast_metrics, policy_metrics, study / "evaluation")
    output.mkdir(parents=True, exist_ok=True)
    for name, table in (("forecast_metrics", forecast_metrics), ("paired_forecast_comparisons", forecast_pairs),
                        ("recommendation_metrics", policy_metrics), ("paired_policy_comparisons", policy_pairs)):
        table.to_csv(output / f"{name}.csv", index=False, float_format="%.10g")
    audit = {"analysis_type": "Retrospective exploratory subgroup supplement; no retraining or weight selection",
             "source_study": study.name, "test_start": test["decision_time"].min().isoformat(),
             "test_end": test["decision_time"].max().isoformat(),
             "subgroups": {"overall": "All matched held-out observations", "evening_peak": "17:00 <= departure < 20:00 Singapore time",
                           "weekend": "Departure weekday is Saturday or Sunday"},
             "groups_may_overlap": True, "forecast_support": "All B0/B1/M0/M1 finite, valid origin and same-segment valid arrival",
             "policy_support": "Existing matched candidate sets and observed endpoints, separately by candidate cohort",
             "primary_event": "full at arrival: 0 spaces", "secondary_events": ["low spaces: <3", "near-full: <10% free, including full"],
             "selected_distance_weight": settings["selected_distance_weight"], "low_spaces_threshold": 3,
             "overall_published_metrics_verified": True, "bootstrap_repetitions": repetitions, "seed": seed,
             "intervals": "95% percentile paired day-block bootstrap; intervals not multiplicity-adjusted",
             "tests": "Two-sided paired day sign flips; Holm across all available comparisons in each CSV",
             "assumptions": "Independent, exchangeable date blocks; symmetric day effects under the sign-flip null. Adjacent-day dependence is not modelled.",
             "interpretation": "Subgroup gains are descriptive across groups; no interaction test establishes that one subgroup gains more than another. No interval or p-value for no_day_level_variation does not establish equivalence."}
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if figures:
        make_figures(forecast_metrics, policy_metrics, output)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=ROOT / "outputs/history_comparison/two_year_standard")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/rq_supplement/subgroups")
    parser.add_argument("--repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_261_105)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.study_dir, args.output, repetitions=args.repetitions, seed=args.seed,
                         figures=not args.no_figures), indent=2))


if __name__ == "__main__":
    main()
