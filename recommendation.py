"""Part B: distance-aware ranking and matched historical replay."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["decision_time", "horizon_min", "carpark_id"]
CASE = KEY[:2]
SCHEDULES = {"WHOLE DAY": (0, 1440), "NO": (0, 0), "7AM-10.30PM": (420, 1350),
             "7AM-7PM": (420, 1140)}
BASE_STRATEGIES = ("A", "B", "C0", "C")
RISK_SIGNALS = {"C_XGB_RISK": "p_full_xgboost", "C_CAT_RISK": "p_full_catboost"}
SIGNALS = {"C0": "current_occupancy", "C": "forecast_occupancy", **RISK_SIGNALS}
OBSERVATIONS = ["current_is_valid", "current_lots_available", "current_occupancy", "total_lots",
                "actual_is_observed", "actual_lots_available", "actual_total_lots"]
DEFAULT_WEIGHTS = (0.25, 0.5, 0.75)


def visitor_open(short_term_parking, arrival_time):
    """Check short-term visitor access at the Singapore arrival time."""
    try:
        start, end = SCHEDULES[short_term_parking]
    except KeyError as error:
        raise ValueError(f"Unsupported short-term schedule: {short_term_parking}") from error
    local = pd.Timestamp(arrival_time)
    if local.tzinfo is not None:
        local = local.tz_convert("Asia/Singapore")
    return start <= local.hour * 60 + local.minute < end


def rank_candidates(candidates, strategy, distance_weight, *, by=()):
    """Sort from best to worst using only departure-time fields and distance."""
    if not math.isfinite(distance_weight) or not 0 <= distance_weight <= 1:
        raise ValueError("Distance weight must be in [0, 1]")
    if strategy == "A":
        score = candidates["distance_normalized"]
    elif strategy == "B":
        score = -candidates["current_lots_available"]
    elif strategy in SIGNALS:
        score = distance_weight * candidates["distance_normalized"] + (1 - distance_weight) * candidates[SIGNALS[strategy]]
    else:
        raise ValueError(f"Unknown recommendation strategy: {strategy}")
    return candidates.assign(score=score).sort_values([*by, "score", "distance_m", "carpark_id"], kind="stable")


def read_carparks(path, destination):
    parks = pd.read_csv(path, dtype={"carpark_id": "string"})
    required = {"carpark_id", "address", "x_coord", "y_coord", "short_term_parking", "night_parking"}
    if missing := required.difference(parks.columns):
        raise ValueError(f"Missing carpark metadata columns: {', '.join(sorted(missing))}")
    if parks["carpark_id"].isna().any() or parks["carpark_id"].duplicated().any():
        raise ValueError("Carpark metadata must have unique, nonmissing IDs")
    x, y, radius = (float(destination[k]) for k in ("x_coord", "y_coord", "radius_m"))
    if not all(map(math.isfinite, (x, y, radius))) or radius <= 0:
        raise ValueError("Destination must have finite SVY21 coordinates and a positive radius")
    for col in ("x_coord", "y_coord"):
        parks[col] = pd.to_numeric(parks[col], errors="coerce")
        if not np.isfinite(parks[col]).all():
            raise ValueError("All carparks must have finite SVY21 coordinates")
    if not parks["short_term_parking"].isin(SCHEDULES).all():
        raise ValueError("Unrecognized short-term parking schedule; verify it before replay")
    if not parks["night_parking"].isin(["YES", "NO"]).all():
        raise ValueError("Unrecognized night parking flag; verify it before replay")
    if (parks["short_term_parking"].eq("WHOLE DAY") & parks["night_parking"].ne("YES")).any():
        raise ValueError("Whole-day short-term schedule conflicts with night parking flag")
    parks["distance_m"] = np.hypot(parks["x_coord"] - x, parks["y_coord"] - y)
    parks["distance_normalized"] = parks["distance_m"] / radius
    parks["in_radius"] = parks["distance_m"].le(radius)
    parks["pilot_candidate"] = parks["in_radius"] & parks["short_term_parking"].ne("NO")
    return parks.sort_values(["distance_m", "carpark_id"])


def read_predictions(path, required):
    frame = pd.read_csv(path, dtype={"carpark_id": "string"})
    if missing := set([*KEY, "target_time", *required]).difference(frame.columns):
        raise ValueError(f"Missing prediction columns: {', '.join(sorted(missing))}")
    for col in ("decision_time", "target_time"):
        frame[col] = pd.to_datetime(frame[col], format="ISO8601")
        if frame[col].isna().any() or frame[col].dt.tz is not None:
            raise ValueError("Replay expects naive Singapore local timestamps without missing values")
    if not frame["horizon_min"].isin([15, 30]).all():
        raise ValueError("Replay expects 15- or 30-minute forecasts")
    if not (frame["target_time"] - frame["decision_time"]).eq(pd.to_timedelta(frame["horizon_min"], unit="min")).all():
        raise ValueError("Every target_time must equal decision_time plus horizon_min")
    if frame.duplicated(KEY).any() or frame["carpark_id"].isna().any():
        raise ValueError("Predictions require unique, nonmissing decision/horizon/carpark keys")
    return frame


def join_risk(predictions, risk):
    risk_columns = list(RISK_SIGNALS.values())
    matched = predictions.merge(risk, on=KEY, suffixes=("", "_risk"), how="inner", validate="one_to_one")
    if matched.empty or not matched["split"].eq("test").all():
        raise ValueError("Risk predictions must overlap occupancy test keys only")
    for col in ["target_time", *OBSERVATIONS]:
        first, second = matched[col], matched[f"{col}_risk"]
        if not (first.eq(second) | (first.isna() & second.isna())).fillna(False).all():
            raise ValueError(f"Occupancy and risk forecasts disagree on observed {col}")
    return predictions.merge(risk[[*KEY, *risk_columns]], on=KEY, how="left", validate="one_to_one")


def matched_cases(predictions, carparks, *, risk=False):
    """Build the expected visitor-eligible pool, then audit complete cases."""
    origins = predictions[[*CASE, "target_time", "split"]].drop_duplicates()
    expected = origins.merge(carparks.loc[carparks["pilot_candidate"]], how="cross")
    minute = expected["target_time"].dt.hour * 60 + expected["target_time"].dt.minute
    opens = expected["short_term_parking"].map({key: t[0] for key, t in SCHEDULES.items()})
    closes = expected["short_term_parking"].map({key: t[1] for key, t in SCHEDULES.items()})
    expected = expected.loc[minute.ge(opens) & minute.lt(closes)]
    value_columns = [*OBSERVATIONS, "forecast_occupancy", *(RISK_SIGNALS.values() if risk else ())]
    candidates = expected.merge(predictions[[*KEY, *value_columns]], on=KEY, how="left",
                                validate="one_to_one", indicator=True)
    candidates["missing_origin"] = candidates["_merge"].ne("both")
    present = ~candidates["missing_origin"]
    current = candidates["current_lots_available"]
    arrival = candidates["actual_lots_available"]
    current_capacity, arrival_capacity = candidates["total_lots"], candidates["actual_total_lots"]
    valid_current = (candidates["current_is_valid"].eq(True) & candidates["current_occupancy"].between(0, 1)
                     & np.isfinite(current_capacity) & current_capacity.gt(0) & current.ge(0)
                     & current.le(current_capacity))
    valid_arrival = (candidates["actual_is_observed"].eq(True) & np.isfinite(arrival_capacity)
                     & arrival_capacity.gt(0) & arrival.ge(0) & arrival.le(arrival_capacity))
    forecast_cols = ["forecast_occupancy", *(RISK_SIGNALS.values() if risk else ())]
    valid_forecast = candidates[forecast_cols].apply(lambda col: col.between(0, 1)).all(axis=1)
    candidates["invalid_departure"] = present & ~valid_current
    candidates["missing_arrival"] = present & ~valid_arrival
    candidates["missing_forecast"] = present & ~valid_forecast
    reasons = ["missing_origin", "invalid_departure", "missing_arrival", "missing_forecast"]
    grouped = candidates.groupby(CASE, sort=False)
    stats = grouped[reasons].sum().join(grouped.size().rename("expected_candidates"))
    audit = origins.merge(stats, on=CASE, how="left", validate="one_to_one")
    audit[[*reasons, "expected_candidates"]] = audit[[*reasons, "expected_candidates"]].fillna(0).astype(int)
    audit["complete"] = audit["expected_candidates"].gt(0) & audit[reasons].sum(axis=1).eq(0)
    candidates = candidates.merge(audit[[*CASE, "complete"]], on=CASE, validate="many_to_one")
    return candidates.loc[candidates["complete"]].copy(), audit


def evaluate(candidates, strategies, weight, low_spaces_threshold, cohort):
    """Rank first, then score picks against observed arrival outcomes."""
    if cohort == "open_at_departure":
        candidates = candidates.loc[candidates["current_lots_available"].gt(0)]
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame()
    nearest = candidates.groupby(CASE)["distance_m"].min().rename("nearest_distance_m")
    count = candidates.groupby(CASE).size().rename("candidate_count")
    outcomes, rankings = [], []
    for strategy in strategies:
        ranked = rank_candidates(candidates, strategy, weight, by=CASE)
        ranked["rank"] = ranked.groupby(CASE, sort=False).cumcount() + 1
        top = ranked.loc[ranked["rank"].le(3)].copy()
        top["strategy"], top["distance_weight"], top["cohort"] = strategy, weight, cohort
        top = top.join(nearest, on=CASE).join(count, on=CASE)
        top["extra_distance_m"] = top["distance_m"] - top["nearest_distance_m"]
        top["full_at_arrival"] = top["actual_lots_available"].eq(0)
        top["near_full_at_arrival"] = top["actual_lots_available"].mul(10).lt(top["actual_total_lots"])
        top["low_spaces_at_arrival"] = top["actual_lots_available"].lt(low_spaces_threshold)
        any_available = (~top["full_at_arrival"]).groupby([top[k] for k in CASE]).any().rename("top3_any_available")
        first = top.loc[top["rank"].eq(1)].join(any_available, on=CASE)
        for horizon, rows in first.groupby("horizon_min", sort=True):
            outcomes.append({
                "horizon_min": horizon, "cohort": cohort, "strategy": strategy, "distance_weight": weight,
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
        rankings.append(top)
    return pd.DataFrame(outcomes), pd.concat(rankings, ignore_index=True)


def select_weight(validation, weights, low_spaces_threshold):
    """Choose a shared weight using C0 validation only, preferring 0.5 on ties."""
    if validation.empty:
        raise ValueError("No complete validation cases; cannot select a distance weight")
    summaries = [evaluate(validation, ["C0"], w, low_spaces_threshold, "visitor_eligible")[0]
                 for w in weights]
    by_weight = pd.concat(summaries, ignore_index=True)
    totals = by_weight.groupby("distance_weight")["top1_full_count"].sum()
    selected = min(weights, key=lambda w: (totals.loc[w], abs(w - 0.5), w))
    by_weight["selected"] = by_weight["distance_weight"].eq(selected)
    return selected, by_weight


def paired_differences(rankings, comparisons):
    first = rankings.loc[rankings["rank"].eq(1)]
    key = [*CASE, "cohort", "distance_weight"]
    metrics = ["full_at_arrival", "near_full_at_arrival", "low_spaces_at_arrival", "distance_m"]
    baseline = first.loc[first["strategy"].eq("C0"), [*key, *metrics]]
    reports = []
    for strategy in comparisons:
        alternative = first.loc[first["strategy"].eq(strategy), [*key, *metrics]]
        paired = alternative.merge(baseline, on=key, suffixes=("", "_c0"), validate="one_to_one")
        if len(paired) != len(baseline) or len(paired) != len(alternative):
            raise ValueError("Strategies must be compared on identical cases")
        for keys, rows in paired.groupby(["horizon_min", "cohort", "distance_weight"], sort=True):
            horizon, cohort, weight = keys
            event_delta = rows["full_at_arrival"].astype(int) - rows["full_at_arrival_c0"].astype(int)
            reports.append({
                "comparison": f"{strategy}-C0", "horizon_min": horizon, "cohort": cohort,
                "distance_weight": weight, "matched_cases": len(rows),
                "full_rate_delta": event_delta.mean(), "full_avoided": int(event_delta.lt(0).sum()),
                "new_full": int(event_delta.gt(0).sum()),
                "near_full_rate_delta": rows["near_full_at_arrival"].mean() - rows["near_full_at_arrival_c0"].mean(),
                "low_spaces_rate_delta": rows["low_spaces_at_arrival"].mean() - rows["low_spaces_at_arrival_c0"].mean(),
                "mean_distance_delta_m": (rows["distance_m"] - rows["distance_m_c0"]).mean(),
            })
    return pd.DataFrame(reports)


def run(forecasts, metadata, destination_path, output_dir, *, risk_path=None, weights=DEFAULT_WEIGHTS,
        forecast_column="M1", low_spaces_threshold=3):
    weights = tuple(sorted(set(weights)))
    if not weights or any(not math.isfinite(w) or not 0 <= w <= 1 for w in weights):
        raise ValueError("At least one valid distance weight in [0, 1] is required")
    if not isinstance(low_spaces_threshold, int) or low_spaces_threshold < 1:
        raise ValueError("Low-space threshold must be a positive integer")
    destination = json.loads(Path(destination_path).read_text(encoding="utf-8"))
    parks = read_carparks(metadata, destination)
    predictions = read_predictions(forecasts, ["split", forecast_column, *OBSERVATIONS])
    if set(predictions["split"].unique()) != {"validation", "test"}:
        raise ValueError("Occupancy forecasts must contain both validation and test splits only")
    val_mask = predictions["split"].eq("validation")
    if predictions.loc[val_mask, "target_time"].max() >= predictions.loc[~val_mask, "decision_time"].min():
        raise ValueError("Validation outcomes must end before the first test decision")
    predictions["forecast_occupancy"] = predictions[forecast_column]
    has_risk = risk_path is not None
    if has_risk:
        risk = read_predictions(risk_path, ["low_spaces_threshold", *OBSERVATIONS, *RISK_SIGNALS.values()])
        if not risk["low_spaces_threshold"].eq(low_spaces_threshold).all():
            raise ValueError("Recommendation and risk forecasts must use the same low-space threshold")
        predictions = join_risk(predictions, risk)
    validation, val_audit = matched_cases(predictions.loc[val_mask], parks)
    test, test_audit = matched_cases(predictions.loc[~val_mask], parks, risk=has_risk)
    weight, validation_results = select_weight(validation, weights, low_spaces_threshold)
    if test.empty:
        raise ValueError("No complete test cases; verify carpark cohort, forecasts and arrival records")
    strategies = [*BASE_STRATEGIES, *(RISK_SIGNALS if has_risk else ())]
    sensitivity, selected_rankings = [], []
    for w in weights:
        for cohort in ("visitor_eligible", "open_at_departure"):
            summary, ranked = evaluate(test, strategies, w, low_spaces_threshold, cohort)
            if not summary.empty:
                sensitivity.append(summary)
                if w == weight:
                    selected_rankings.append(ranked)
    sensitivity = pd.concat(sensitivity, ignore_index=True)
    rankings = pd.concat(selected_rankings, ignore_index=True)
    primary = sensitivity.loc[sensitivity["distance_weight"].eq(weight)]
    comparison = paired_differences(rankings, [s for s in strategies if s not in ("A", "B", "C0")])
    audit = pd.concat([val_audit, test_audit], ignore_index=True)
    ranking_columns = [*KEY, "target_time", "cohort", "strategy", "rank", "candidate_count", "distance_weight",
                       "score", "distance_m", "extra_distance_m", "current_lots_available", "current_occupancy",
                       "forecast_occupancy", *(RISK_SIGNALS.values() if has_risk else ()),
                       "actual_lots_available", "actual_total_lots", "full_at_arrival", "near_full_at_arrival",
                       "low_spaces_at_arrival"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {"recommendation_summary": primary, "recommendation_rankings": rankings[ranking_columns],
             "weight_sensitivity": sensitivity, "validation_weights": validation_results,
             "paired_differences": comparison, "replay_case_audit": audit, "carpark_distances": parks}
    for name, frame in files.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, date_format="%Y-%m-%dT%H:%M:%S", float_format="%.10g")
    settings = {"destination": destination, "selected_distance_weight": weight, "weight_grid": weights,
                "forecast_column": forecast_column, "low_spaces_threshold": low_spaces_threshold,
                "risk_strategies_included": has_risk}
    (output_dir / "replay_settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecasts", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--risk", type=Path, help="Optional direct risk forecasts for matched test comparisons")
    parser.add_argument("--weights", type=float, nargs="+", default=DEFAULT_WEIGHTS)
    parser.add_argument("--forecast-column", choices=["M0", "M1"], default="M1")
    parser.add_argument("--low-spaces-threshold", type=int, default=3)
    args = parser.parse_args()
    try:
        settings = run(args.forecasts, args.metadata, args.destination, args.output, risk_path=args.risk,
                       weights=args.weights, forecast_column=args.forecast_column,
                       low_spaces_threshold=args.low_spaces_threshold)
    except ValueError as error:
        parser.error(str(error))
    print(f"Replay complete; shared validation-selected weight = {settings['selected_distance_weight']:.2f}")


if __name__ == "__main__":
    main()
