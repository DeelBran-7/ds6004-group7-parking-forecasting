"""Compare one- and two-year training histories on identical held-out dates.

The inputs are completed A/B/C study directories. Scores are recomputed from
paired observations rather than inferred from independently averaged summaries.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate import read_inputs, require_same_values
from evaluation_stats import holm_adjust, paired_day_inference


KEY = ["decision_time", "horizon_min", "carpark_id"]
POLICY_KEY = ["decision_time", "horizon_min", "cohort", "strategy"]
OBSERVATIONS = ["target_time", "current_is_valid", "current_lots_available", "total_lots",
                "actual_is_observed", "actual_lots_available", "actual_total_lots", "forecast_label_eligible"]
FORECAST_MODELS = ("B0", "B1", "M0", "M1")
RISK_MODELS = ("persistence", "occupancy_proxy", "xgboost", "catboost")
POLICY_METRICS = {"full_rate": "full_at_arrival", "near_full_rate": "near_full_at_arrival",
                  "low_spaces_rate": "low_spaces_at_arrival", "mean_distance_m": "distance_m"}


def align(left, right, keys, label):
    """Order identical observation keys without silently dropping a case."""
    for frame in (left, right):
        if frame.empty or frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError(f"{label} needs nonempty unique keys")
    left = left.set_index(keys).sort_index()
    right = right.set_index(keys).sort_index()
    if not left.index.equals(right.index):
        raise ValueError(f"{label} must have identical keys across history lengths")
    return left.reset_index(), right.reset_index()


def same_fields(left, right, fields, label):
    for column in fields:
        require_same_values(left[column], right[column], f"{label} differs on {column}")


def valid_labels(frame):
    keep = (frame["current_is_valid"].eq(True) & frame["actual_is_observed"].eq(True)
            & frame["forecast_label_eligible"].eq(True))
    free = frame.loc[keep, "actual_lots_available"]
    capacity = frame.loc[keep, "actual_total_lots"]
    if not (np.isfinite(capacity) & capacity.gt(0) & np.isfinite(free) & free.between(0, capacity)).all():
        raise ValueError("Eligible model rows contain invalid arrival counts")
    return keep


def common_predictions(left, right, columns):
    """Require shared finite predictions across all comparators and histories."""
    common = pd.Series(True, index=left.index)
    for frame in (left, right):
        probabilities = frame[list(columns)]
        finite = np.isfinite(probabilities)
        in_range = probabilities.ge(0) & probabilities.le(1)
        if (finite & ~in_range).any().any():
            raise ValueError("Finite predictions must lie in [0, 1]")
        if not finite.any().all():
            raise ValueError("The history comparison requires every core model to have predictions")
        common &= finite.all(axis=1)
    return common


def comparison(days, one, two, repetitions, seed):
    stats = paired_day_inference(days, two, one, repetitions=repetitions, seed=seed)
    one_mean, two_mean = stats["reference_mean"], stats["candidate_mean"]
    reduction = ((one_mean - two_mean) / one_mean
                 if one_mean is not None and one_mean > 0 else None)
    return {"one_year_score": one_mean, "two_year_score": two_mean,
            "relative_reduction": reduction, **stats}


def correct_family(records):
    frame = pd.DataFrame(records)
    values = [float(p) if pd.notna(p) else None for p in frame["p_value"]]
    frame["p_value_holm"] = holm_adjust(values)
    return frame


def compare_forecasts(one, two, repetitions, seed):
    eligible = valid_labels(one) & valid_labels(two)
    common = common_predictions(one, two, FORECAST_MODELS)
    observed = 1 - one["actual_lots_available"] / one["actual_total_lots"]
    near = one["actual_lots_available"].mul(10).lt(one["actual_total_lots"])
    cohorts = {"all": pd.Series(True, index=one.index), "near_full": near,
               "holiday": one["is_holiday"].eq(1), "non_holiday": one["is_holiday"].eq(0)}
    records = []
    for horizon in (15, 30):
        horizon_rows = one["horizon_min"].eq(horizon)
        for cohort, condition in cohorts.items():
            keep = horizon_rows & eligible & common & condition
            days = one.loc[keep, "decision_time"].dt.normalize()
            for model in FORECAST_MODELS:
                first_loss = (one.loc[keep, model] - observed.loc[keep]).abs()
                second_loss = (two.loc[keep, model] - observed.loc[keep]).abs()
                records.append({"horizon_min": horizon, "cohort": cohort, "model": model,
                                "metric": "mae", "n_eligible": int((horizon_rows & eligible & condition).sum()),
                                **comparison(days, first_loss, second_loss, repetitions, seed)})
    return correct_family(records)


def compare_risk(one, two, repetitions, seed):
    eligible = valid_labels(one) & valid_labels(two)
    threshold_values = one["low_spaces_threshold"].unique()
    if len(threshold_values) != 1 or threshold_values[0] < 1 or threshold_values[0] % 1:
        raise ValueError("Risk comparison requires one positive integer low-space threshold")
    targets = {"full": one["actual_lots_available"].eq(0),
               "low_spaces": one["actual_lots_available"].lt(threshold_values[0])}
    cohorts = {"all": pd.Series(True, index=one.index),
               "open_at_departure": one["current_lots_available"].gt(0)}
    records = []
    for event, truth in targets.items():
        columns = [f"p_{event}_{model}" for model in RISK_MODELS]
        common = common_predictions(one, two, columns)
        for horizon in (15, 30):
            horizon_rows = one["horizon_min"].eq(horizon)
            for cohort, condition in cohorts.items():
                keep = horizon_rows & eligible & common & condition
                days = one.loc[keep, "decision_time"].dt.normalize()
                label = truth.loc[keep].astype(float)
                for model in RISK_MODELS:
                    column = f"p_{event}_{model}"
                    first_loss = (one.loc[keep, column] - label).pow(2)
                    second_loss = (two.loc[keep, column] - label).pow(2)
                    records.append({"horizon_min": horizon, "event": event, "cohort": cohort,
                                    "model": model, "metric": "brier", "events": int(label.sum()),
                                    "n_eligible": int((horizon_rows & eligible & condition).sum()),
                                    **comparison(days, first_loss, second_loss, repetitions, seed)})
    return correct_family(records)


def read_first_choices(path):
    columns = {*POLICY_KEY, "carpark_id", "target_time", "rank", "candidate_count", "distance_weight",
               "actual_lots_available", "actual_total_lots", *POLICY_METRICS.values()}
    selected = []
    for chunk in pd.read_csv(path, usecols=columns.__contains__, dtype={"carpark_id": "string"}, chunksize=100_000):
        selected.append(chunk.loc[chunk["rank"].eq(1)])
    result = pd.concat(selected, ignore_index=True)
    for column in ("decision_time", "target_time"):
        result[column] = pd.to_datetime(result[column], format="ISO8601")
    return result


def check_choices(choices, forecast, threshold):
    source = forecast[[*KEY, "target_time", "actual_lots_available", "actual_total_lots"]]
    checked = choices.merge(source, on=KEY, how="left", validate="many_to_one", suffixes=("", "_source"), indicator=True)
    if not checked["_merge"].eq("both").all():
        raise ValueError("A ranked first choice is absent from the held-out forecast rows")
    for column in ("target_time", "actual_lots_available", "actual_total_lots"):
        require_same_values(checked[column], checked[f"{column}_source"], f"Ranked outcomes disagree on {column}")
    free, capacity = checked["actual_lots_available"], checked["actual_total_lots"]
    if not (np.isfinite(free) & np.isfinite(capacity) & capacity.gt(0) & free.between(0, capacity)).all():
        raise ValueError("A ranked arrival has invalid counts")
    labels = {"full_at_arrival": free.eq(0), "low_spaces_at_arrival": free.lt(threshold),
              "near_full_at_arrival": free.mul(10).lt(capacity)}
    for column, observed in labels.items():
        require_same_values(checked[column], observed, f"Ranked event label disagrees on {column}")
    if not (np.isfinite(choices["distance_m"]) & choices["distance_m"].ge(0)).all():
        raise ValueError("First-choice distances must be finite and nonnegative")


def compare_policies(one, two, repetitions, seed):
    records = []
    for (horizon, cohort, strategy), first in one.groupby(["horizon_min", "cohort", "strategy"], sort=True):
        second = two.loc[first.index]
        changes = int(first["carpark_id"].ne(second["carpark_id"]).sum())
        days = first["decision_time"].dt.normalize()
        for metric, column in POLICY_METRICS.items():
            records.append({"horizon_min": horizon, "cohort": cohort, "strategy": strategy,
                            "metric": metric, "changed_first_choices": changes,
                            "one_year_events": int(first[column].sum()) if metric != "mean_distance_m" else None,
                            "two_year_events": int(second[column].sum()) if metric != "mean_distance_m" else None,
                            **comparison(days, first[column], second[column], repetitions, seed)})
    return correct_family(records)


def plot_scores(frame, path, *, risk=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = RISK_MODELS if risk else FORECAST_MODELS
    labels = (["Persistence", "Occupancy proxy", "XGBoost risk", "CatBoost risk"] if risk
              else ["B0 Current occupancy", "B1 Previous week", "M0 XGBoost", "M1 XGBoost + holidays"])
    selected = frame.loc[frame["cohort"].eq("open_at_departure" if risk else "all")]
    if risk:
        selected = selected.loc[selected["event"].eq("full")]
    scale = 1 if risk else 100
    upper = selected[["one_year_score", "two_year_score"]].max().max() * scale
    fig, axes = plt.subplots(1, 2, figsize=(11.1, 5.1))
    fig.subplots_adjust(left=.245, right=.97, top=.72, bottom=.22, wspace=.26)
    fig.text(.045, .955, "One-year and two-year history on the same test dates", fontsize=17,
             fontweight="bold", color="#172D3D", va="top")
    subtitle = ("Full-at-arrival Brier score | open at departure | lower is better" if risk
                else "Occupancy MAE | identical model settings and shared observations | lower is better")
    fig.text(.045, .893, subtitle, fontsize=10.8, color="#506478", va="top")
    for horizon, ax in zip((15, 30), axes):
        data = selected.loc[selected["horizon_min"].eq(horizon)].set_index("model").loc[list(models)]
        if data["n_pairs"].nunique() != 1:
            raise ValueError("Figure needs shared sample counts across models")
        y = np.arange(len(models))
        for column, offset, color, label in (("one_year_score", -.17, "#63768C", "One-year history"),
                                             ("two_year_score", .17, "#006DA3", "Two-year history")):
            values = data[column] * scale
            ax.barh(y + offset, values, height=.29, color=color, label=label)
            for position, value in zip(y + offset, values):
                text = f"{value:.5f}" if risk else f"{value:.3f}"
                ax.text(value + upper*.022, position, text, va="center", fontsize=8.2, color="#172D3D")
        ax.set_title(f"{horizon} minutes | {int(data['n_pairs'].iloc[0]):,} paired rows", loc="left",
                     fontsize=10.5, color="#172D3D", fontweight="bold", pad=12)
        ax.set(yticks=y, yticklabels=labels if horizon == 15 else [], ylim=(3.55, -.55),
               xlim=(0, upper*1.29), xlabel="Brier score" if risk else "MAE (percentage points)")
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#DDE6ED", linewidth=.7)
        ax.tick_params(length=0, colors="#506478", labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.55, .83), ncols=2,
               frameon=False, fontsize=10)
    fig.text(.045, .075, "Only the available training history changes; validation and test dates are held fixed.",
             fontsize=9.2, color="#506478")
    fig.text(.045, .035, "Bars are point estimates. Paired day-bootstrap intervals and Holm-adjusted tests are in the accompanying CSV.",
             fontsize=8.7, color="#506478")
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def run(one_dir, two_dir, output, *, repetitions=10_000, seed=20_261_105, figures=True):
    one_dir, two_dir, output = map(Path, (one_dir, two_dir, output))
    first = read_inputs(one_dir / "forecast/predictions.csv")
    second = read_inputs(two_dir / "forecast/predictions.csv")
    first, second = align(first, second, KEY, "Validation/test forecasts")
    same_fields(first, second, [*OBSERVATIONS, "split", "is_holiday", "B0", "B1", "actual_occupancy"], "Forecast inputs")
    if set(first["split"]) != {"validation", "test"}:
        raise ValueError("Forecast files must contain validation and test splits")
    validation = first.loc[first["split"].eq("validation")]
    one_test = first.loc[first["split"].eq("test")].reset_index(drop=True)
    two_test = second.loc[second["split"].eq("test")].reset_index(drop=True)
    if validation["target_time"].max() >= one_test["decision_time"].min():
        raise ValueError("Validation targets overlap test decisions")
    forecast_results = compare_forecasts(one_test, two_test, repetitions, seed)

    one_risk = read_inputs(one_dir / "risk/risk_predictions.csv")
    two_risk = read_inputs(two_dir / "risk/risk_predictions.csv")
    one_risk, two_risk = align(one_risk, two_risk, KEY, "Risk predictions")
    same_fields(one_risk, two_risk, [*OBSERVATIONS, "low_spaces_threshold"], "Risk inputs")
    for event in ("full", "low_spaces"):
        same_fields(one_risk, two_risk, [f"p_{event}_persistence"], "Calibrated persistence baseline")
    for forecast, risk in ((one_test, one_risk), (two_test, two_risk)):
        forecast, risk = align(forecast, risk, KEY, "Forecast/risk test rows")
        same_fields(forecast, risk, OBSERVATIONS, "Forecast/risk observations")
    risk_results = compare_risk(one_risk, two_risk, repetitions, seed)

    settings = [json.loads((path / "recommendation/replay_settings.json").read_text()) for path in (one_dir, two_dir)]
    for field in ("destination", "selected_distance_weight", "weight_grid", "forecast_column", "low_spaces_threshold"):
        if settings[0][field] != settings[1][field]:
            raise ValueError(f"Recommendation configurations differ on {field}")
    one_choices = read_first_choices(one_dir / "recommendation/recommendation_rankings.csv")
    two_choices = read_first_choices(two_dir / "recommendation/recommendation_rankings.csv")
    one_choices, two_choices = align(one_choices, two_choices, POLICY_KEY, "Recommended first-choice cases")
    same_fields(one_choices, two_choices, ["target_time", "candidate_count", "distance_weight"], "Recommendation cases")
    for choices, forecast in ((one_choices, one_test), (two_choices, two_test)):
        check_choices(choices, forecast, settings[0]["low_spaces_threshold"])
    baseline = one_choices["strategy"].isin(["A", "B", "C0"])
    same_fields(one_choices.loc[baseline], two_choices.loc[baseline],
                ["carpark_id", *POLICY_METRICS.values()], "History-independent recommendation baselines")
    policy_results = compare_policies(one_choices, two_choices, repetitions, seed)

    output.mkdir(parents=True, exist_ok=True)
    tables = {"forecast_history_comparison": forecast_results, "risk_history_comparison": risk_results,
              "policy_history_comparison": policy_results}
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False, float_format="%.10g")
    audit = {
        "comparison": "two_year_shared minus one_year", "test_start": one_test["decision_time"].min().isoformat(),
        "test_end": one_test["decision_time"].max().isoformat(),
        "validation_start": validation["decision_time"].min().isoformat(),
        "validation_last_target": validation["target_time"].max().isoformat(),
        "test_dates": int(one_test["decision_time"].dt.normalize().nunique()),
        "identical_forecast_test_rows": len(one_test), "identical_first_choice_cases": len(one_choices),
        "selected_distance_weight": settings[0]["selected_distance_weight"],
        "low_spaces_threshold": settings[0]["low_spaces_threshold"],
        "repetitions": repetitions, "seed": seed,
        "support": "All listed models and both history lengths must have finite forecasts on each scored pair",
        "intervals": "95% paired day-block percentile bootstrap; intervals are not multiplicity-adjusted",
        "tests": "Two-sided paired day sign-flips; Holm correction within each comparison CSV",
        "assumptions": "Exchangeable date blocks and symmetric independent day effects; neighboring-day dependence is not modelled",
        "score_units": "MAE uses occupancy ratios; Brier uses squared probability error; policy rates are fractions; distance uses metres",
        "zero_differences": "No interval or p-value for no_day_level_variation; this does not establish equivalence",
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if figures:
        plot_scores(forecast_results, output / "forecast_history_comparison.png")
        plot_scores(risk_results, output / "full_risk_history_comparison.png", risk=True)
    return audit


def main():
    base = ROOT / "outputs/history_comparison"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one-year", type=Path, default=base / "one_year")
    parser.add_argument("--two-year", type=Path, default=base / "two_year_shared")
    parser.add_argument("--output", type=Path, default=base / "comparison")
    parser.add_argument("--repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_261_105)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    audit = run(args.one_year, args.two_year, args.output, repetitions=args.repetitions,
                seed=args.seed, figures=not args.no_figures)
    print(f"History comparison complete on {audit['test_dates']} shared test dates: {args.output}")


if __name__ == "__main__":
    main()
