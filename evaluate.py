"""Part C: score held-out A/B outputs and compare policies on matched days."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_models import evaluate_forecasts, evaluate_risk
from evaluation_stats import holm_adjust, paired_day_inference


KEY = ["decision_time", "horizon_min", "carpark_id"]
CASE = ["decision_time", "horizon_min", "cohort"]
CHOICE = [*CASE, "strategy"]
CORE = {"A", "B", "C0", "C"}
RISK_POLICIES = {"C_XGB_RISK": "p_full_xgboost", "C_CAT_RISK": "p_full_catboost"}
METRICS = {
    "full_rate": "full_at_arrival", "near_full_rate": "near_full_at_arrival",
    "low_spaces_rate": "low_spaces_at_arrival", "top3_feasibility": "top3_any_available",
    "mean_distance_m": "distance_m",
}
OBSERVATIONS = ["target_time", "current_is_valid", "current_lots_available", "total_lots",
                "actual_is_observed", "actual_lots_available", "actual_total_lots"]
RANKING_FIELDS = [*KEY, "target_time", "cohort", "strategy", "rank", "candidate_count",
                  "distance_weight", "distance_m", "extra_distance_m", "current_lots_available",
                  "forecast_occupancy", "actual_lots_available", "actual_total_lots", *list(METRICS.values())[:3]]


def read_inputs(path, *, usecols=None):
    frame = pd.read_csv(path, usecols=usecols, dtype={"carpark_id": "string"})
    for col in ("decision_time", "target_time"):
        frame[col] = pd.to_datetime(frame[col], format="ISO8601")
        if frame[col].isna().any() or frame[col].dt.tz is not None:
            raise ValueError(f"{Path(path).name} must contain naive Singapore timestamps")
    delta = pd.to_timedelta(frame["horizon_min"], unit="min")
    if not frame["horizon_min"].isin([15, 30]).all() or not frame["target_time"].eq(frame["decision_time"] + delta).all():
        raise ValueError(f"{Path(path).name} has an invalid forecast horizon or arrival timestamp")
    return frame


def require_same_values(left, right, message):
    """Tolerate CSV float formatting, but require the same missing values."""
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        agree = np.isclose(left.to_numpy(dtype=float, na_value=np.nan), right.to_numpy(dtype=float, na_value=np.nan),
                           rtol=1e-8, atol=1e-10, equal_nan=True)
    else:
        agree = (left.eq(right) | (left.isna() & right.isna())).fillna(False)
    if not np.all(agree):
        raise ValueError(message)


def validate_handoff(occupancy, risk, rankings, settings):
    """Check the inputs belong to one held-out run before comparing results."""
    for name, frame in (("occupancy", occupancy), ("risk", risk)):
        if frame.empty or frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
            raise ValueError(f"{name} needs nonempty unique decision/horizon/carpark keys")
    if set(occupancy["split"].unique()) != {"validation", "test"}:
        raise ValueError("Occupancy file must contain the chronological validation and test splits")
    test = occupancy.loc[occupancy["split"].eq("test")].set_index(KEY).sort_index()
    validation_end = occupancy.loc[occupancy["split"].eq("validation"), "target_time"].max()
    if validation_end >= test.index.get_level_values("decision_time").min():
        raise ValueError("Validation arrivals must precede every test decision")
    risk_test = risk.set_index(KEY).sort_index()
    if not test.index.equals(risk_test.index):
        raise ValueError("Occupancy and risk files must have identical test keys")
    fields = [*OBSERVATIONS, *(["forecast_label_eligible"] if "forecast_label_eligible" in test else [])]
    for col in fields:
        require_same_values(test[col], risk_test[col], f"Occupancy and risk disagree on {col}")

    low_threshold = settings["low_spaces_threshold"]
    if type(low_threshold) is not int or low_threshold < 1 or not risk["low_spaces_threshold"].eq(low_threshold).all():
        raise ValueError("Risk and recommendation need the same positive low-spaces threshold")
    if rankings.empty or not rankings["distance_weight"].eq(settings["selected_distance_weight"]).all():
        raise ValueError("Rankings must use the selected validation weight")
    used_strategies = set(rankings["strategy"])
    if not used_strategies.issubset(CORE | set(RISK_POLICIES)) or not CORE.issubset(used_strategies):
        raise ValueError("Rankings must contain A/B/C0/C and only recognized risk strategies")
    if not rankings["cohort"].isin(["visitor_eligible", "open_at_departure"]).all():
        raise ValueError("Unknown recommendation evaluation cohort")

    target = rankings.merge(test.reset_index()[[*KEY, *OBSERVATIONS, settings["forecast_column"]]],
                            on=KEY, how="left", validate="many_to_one", suffixes=("", "_source"), indicator=True)
    if (target["_merge"].ne("both").any() or not target["current_is_valid"].eq(True).all()
            or not target["actual_is_observed"].eq(True).all()):
        raise ValueError("Ranked choices need valid departure and observed arrival records in the test file")
    for col in ("target_time", "current_lots_available", "actual_lots_available", "actual_total_lots"):
        require_same_values(target[col], target[f"{col}_source"], f"Rankings disagree with A on {col}")
    require_same_values(target["forecast_occupancy"], target[settings["forecast_column"]],
                        "Rankings disagree with the selected occupancy model")
    signals = [col for strategy, col in RISK_POLICIES.items() if strategy in used_strategies]
    if signals:
        risk_scores = rankings.merge(risk[[*KEY, *signals]], on=KEY, validate="many_to_one", suffixes=("", "_source"))
        for col in signals:
            if col not in rankings:
                raise ValueError(f"Rankings are missing {col}")
            require_same_values(risk_scores[col], risk_scores[f"{col}_source"], f"Rankings disagree with A on {col}")
    arrival, capacity = target["actual_lots_available"], target["actual_total_lots"]
    if not (np.isfinite(capacity) & capacity.gt(0) & arrival.between(0, capacity)).all():
        raise ValueError("Ranked arrival counts are invalid")
    checks = {"full_at_arrival": arrival.eq(0), "near_full_at_arrival": arrival.mul(10).lt(capacity),
              "low_spaces_at_arrival": arrival.lt(low_threshold)}
    for col, truth in checks.items():
        require_same_values(target[col], truth, f"Rankings contain inconsistent {col} labels")
    if target.loc[target["cohort"].eq("open_at_departure"), "current_lots_available"].le(0).any():
        raise ValueError("Open-at-departure rankings contain a currently full carpark")


def ranked_choices(rankings):
    """Validate top lists and construct one matched first-choice row per strategy."""
    rankings = rankings.drop(columns=["nearest_distance_m", "top3_any_available"], errors="ignore")
    if rankings[[*CHOICE, "carpark_id", "rank", "candidate_count"]].isna().any().any():
        raise ValueError("Recommendation keys, ranks and candidate counts cannot be missing")
    if rankings.duplicated([*CHOICE, "rank"]).any() or rankings.duplicated([*CHOICE, "carpark_id"]).any():
        raise ValueError("A recommendation list repeats a rank or carpark")
    if not rankings["rank"].isin([1, 2, 3]).all():
        raise ValueError("Recommendation ranks must be 1, 2 or 3")
    if not (rankings["candidate_count"].ge(1) & rankings["candidate_count"].mod(1).eq(0)).all():
        raise ValueError("Candidate counts must be positive integers")
    ordered = rankings.sort_values([*CHOICE, "rank"])
    grouped = ordered.groupby(CHOICE, sort=False, observed=True)
    if not ordered["rank"].eq(grouped.cumcount() + 1).all():
        raise ValueError("Recommendation lists must start at rank one and have no rank gaps")
    if not (grouped.size().eq(grouped["candidate_count"].first().clip(upper=3))
            & grouped["candidate_count"].nunique().eq(1)).all():
        raise ValueError("Recommendation lists must contain the available top three")
    feasible = (~ordered["full_at_arrival"]).groupby([ordered[k] for k in CHOICE], observed=True).any()
    first = ordered.loc[ordered["rank"].eq(1)].join(feasible.rename("top3_any_available"), on=CHOICE)
    counts = first.groupby(CASE, observed=True)
    if not (counts["strategy"].nunique().eq(first["strategy"].nunique())
            & counts["candidate_count"].nunique().eq(1)).all():
        raise ValueError("Strategies need identical cases and candidate counts within each cohort")
    if not (np.isfinite(first["distance_m"]) & first["distance_m"].ge(0)).all():
        raise ValueError("Recommendation distances must be finite and nonnegative")
    nearest = first.loc[first["strategy"].eq("A")].set_index(CASE)["distance_m"].rename("nearest_distance_m")
    first = first.join(nearest, on=CASE)
    extra = first["distance_m"] - first["nearest_distance_m"]
    if extra.lt(-1e-6).any():
        raise ValueError("Strategy A is not nearest on at least one case")
    require_same_values(extra, first["extra_distance_m"], "Extra distance disagrees with strategy A")
    return first


def evaluate_recommendations(rankings, *, repetitions, seed):
    first = ranked_choices(rankings)
    summaries, comparisons = [], []
    for (horizon, cohort), subset in first.groupby(["horizon_min", "cohort"], sort=True):
        for strategy, rows in subset.groupby("strategy", sort=False):
            summaries.append({
                "horizon_min": horizon, "cohort": cohort, "strategy": strategy,
                "matched_cases": len(rows), "decision_dates": rows["decision_time"].dt.date.nunique(),
                "top1_full_count": int(rows["full_at_arrival"].sum()),
                "top1_full_rate": rows["full_at_arrival"].mean(),
                "top1_near_full_count": int(rows["near_full_at_arrival"].sum()),
                "top1_near_full_rate": rows["near_full_at_arrival"].mean(),
                "top1_low_spaces_count": int(rows["low_spaces_at_arrival"].sum()),
                "top1_low_spaces_rate": rows["low_spaces_at_arrival"].mean(),
                "top3_any_available_rate": rows["top3_any_available"].mean(),
                "mean_distance_m": rows["distance_m"].mean(),
                "mean_extra_vs_nearest_m": rows["extra_distance_m"].mean(),
            })
        reference = subset.loc[subset["strategy"].eq("C0")].set_index("decision_time").sort_index()
        for strategy in ("C", "C_XGB_RISK", "C_CAT_RISK"):
            alternative = subset.loc[subset["strategy"].eq(strategy)].set_index("decision_time").sort_index()
            if alternative.empty:
                continue
            if not reference.index.equals(alternative.index):
                raise ValueError("Cannot infer a policy difference from unmatched decision times")
            switched = int(alternative["carpark_id"].ne(reference["carpark_id"]).sum())
            for metric, col in METRICS.items():
                values = alternative[col].to_numpy(dtype=float)
                baseline = reference[col].to_numpy(dtype=float)
                stats = paired_day_inference(alternative.index.normalize(), values, baseline,
                                             repetitions=repetitions, seed=seed)
                comparisons.append({
                    "horizon_min": horizon, "cohort": cohort, "comparison": f"{strategy}-C0", "metric": metric,
                    "primary": strategy == "C" and cohort == "visitor_eligible" and metric == "full_rate",
                    "changed_first_choices": switched, "candidate_events": int(values.sum()) if metric != "mean_distance_m" else None,
                    "reference_events": int(baseline.sum()) if metric != "mean_distance_m" else None,
                    **stats,
                })
    return pd.DataFrame(summaries), pd.DataFrame(comparisons)


def correct_family(frame):
    """Apply Holm across all available tests in this output table."""
    frame = frame.copy()
    frame["p_value_holm"] = holm_adjust([float(p) if pd.notna(p) else None for p in frame["p_value"]])
    return frame


def run(forecasts, risk_path, recommendation_dir, output_dir, *, repetitions=10000, seed=20261105, figures=True):
    recommendation_dir = Path(recommendation_dir)
    settings = json.loads((recommendation_dir / "replay_settings.json").read_text(encoding="utf-8"))
    occupancy = read_inputs(forecasts)
    risk = read_inputs(risk_path)
    rank_fields = set(RANKING_FIELDS) | set(RISK_POLICIES.values())
    rankings = read_inputs(recommendation_dir / "recommendation_rankings.csv", usecols=rank_fields.__contains__)
    validate_handoff(occupancy, risk, rankings, settings)
    forecast_metrics, forecast_pairs = evaluate_forecasts(occupancy, repetitions=repetitions, seed=seed)
    per_carpark = []
    for carpark, rows in occupancy.groupby("carpark_id", observed=True):
        metrics, _ = evaluate_forecasts(rows, repetitions=repetitions, seed=seed, include_comparisons=False)
        per_carpark.append(metrics.assign(carpark_id=carpark))
    risk_metrics, reliability, risk_pairs = evaluate_risk(risk, repetitions=repetitions, seed=seed)
    recommendation_metrics, policy_pairs = evaluate_recommendations(rankings, repetitions=repetitions, seed=seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "forecast_metrics": forecast_metrics, "paired_forecast_comparisons": correct_family(forecast_pairs),
        "forecast_metrics_by_carpark": pd.concat(per_carpark, ignore_index=True),
        "risk_probability_metrics": risk_metrics, "risk_reliability": reliability,
        "paired_risk_comparisons": correct_family(risk_pairs), "recommendation_metrics": recommendation_metrics,
        "paired_policy_comparisons": correct_family(policy_pairs),
    }
    for name, frame in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, float_format="%.10g")
    days = occupancy.loc[occupancy["split"].eq("test"), "decision_time"].dt.normalize()
    metadata = {
        "test_start": days.min().date().isoformat(), "test_end": days.max().date().isoformat(),
        "test_days": days.nunique(), "matched_ranked_rows": len(rankings),
        "resampling_unit": "Singapore decision date; retain all paired observations within a date",
        "bootstrap_repetitions": repetitions, "seed": seed,
        "intervals": "95% percentile paired day-block bootstrap; no multiplicity adjustment",
        "tests": "Two-sided paired day sign flips; Holm adjusted within each comparison table",
        "assumptions": "Days are treated as exchangeable bootstrap blocks and independent for sign flips. The sign-flip null assumes symmetric day-level differences. Adjacent-day dependence can limit these approximations.",
        "difference": "Candidate minus reference; negative is lower error, risk, or distance",
        "selected_distance_weight": settings["selected_distance_weight"],
        "low_spaces_threshold": settings["low_spaces_threshold"],
    }
    (output_dir / "evaluation_audit.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if figures:
        from evaluation_figures import render_evaluation

        render_evaluation(output_dir)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecasts", type=Path, required=True)
    parser.add_argument("--risk", type=Path, required=True)
    parser.add_argument("--recommendation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.forecasts, args.risk, args.recommendation_dir, args.output,
                     repetitions=args.repetitions, seed=args.seed, figures=not args.no_figures)
    except ValueError as error:
        parser.error(str(error))
    print(f"Evaluation complete on {result['test_days']} held-out dates: {args.output}")


if __name__ == "__main__":
    main()
