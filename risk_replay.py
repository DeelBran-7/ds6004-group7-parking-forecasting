"""Matched pilot replay of changes between departure and arrival."""

import numpy as np
import pandas as pd


MODEL_NAMES = ("xgboost", "catboost", "persistence", "occupancy_proxy")
TRANSITION_COLUMNS = (
    "horizon_min", "matched_rows", "vacancy_declined_count", "vacancy_declined_rate",
    "open_at_departure_count", "open_to_full_count", "open_to_full_rate",
    "safe_at_departure_count", "safe_to_low_count", "safe_to_low_rate", "low_spaces_threshold",
)
REPLAY_COLUMNS = (
    "horizon_min", "strategy", "matched_origins", "first_full_count", "first_full_rate",
    "first_low_spaces_count", "first_low_spaces_rate", "top_three_any_available_count",
    "top_three_any_available_rate", "low_spaces_threshold",
)
REQUIRED = {
    "decision_time", "horizon_min", "carpark_id", "current_is_valid", "current_lots_available",
    "current_occupancy", "actual_is_observed", "actual_lots_available", "low_spaces_threshold",
}


def _rate(count, denominator):
    return count / denominator if denominator else np.nan


def summarize_replay(predictions, carparks):
    """Score risk-only rankings on identical observed, open-at-departure sets.

    Input must be test predictions with one row per origin, horizon, and carpark.
    A replay origin needs all specified carparks present with valid departure and
    arrival observations and finite scores for every included model. Ranks use
    departure data only; recorded arrival counts score the choices afterward.
    The transitions table also includes valid rows outside complete replay sets.
    """
    if missing := REQUIRED.difference(predictions.columns):
        raise ValueError(f"Missing replay columns: {', '.join(sorted(missing))}")
    parks = frozenset(carparks)
    if not parks:
        raise ValueError("Replay requires at least one carpark")
    if "split" in predictions and not predictions["split"].eq("test").all():
        raise ValueError("Replay input must contain only test predictions")
    rows = predictions.loc[predictions["carpark_id"].isin(parks)].copy()
    key = ["decision_time", "horizon_min", "carpark_id"]
    if rows.duplicated(key).any() or rows[key].isna().any().any():
        raise ValueError("Replay input requires unique, nonmissing origin/horizon/carpark keys")
    if rows.empty:
        return pd.DataFrame(columns=TRANSITION_COLUMNS), pd.DataFrame(columns=REPLAY_COLUMNS)
    thresholds = rows["low_spaces_threshold"].unique()
    if len(thresholds) != 1 or pd.isna(thresholds[0]):
        raise ValueError("Replay requires a single low-spaces threshold")
    threshold = float(thresholds[0])
    if not np.isfinite(threshold) or threshold < 1 or not threshold.is_integer():
        raise ValueError("Replay requires a positive integer low-spaces threshold")
    threshold = int(threshold)

    strategies = {
        "most_current_free": (["current_lots_available"], [False]),
        "lowest_current_occupancy": (["current_occupancy"], [True]),
    }
    score_columns = []
    for model in MODEL_NAMES:
        score_pair = [f"p_full_{model}", f"p_low_spaces_{model}"]
        present = [name in rows for name in score_pair]
        if any(present) and not all(present):
            raise ValueError(f"Replay requires both risk scores for {model}")
        if all(present):
            strategies[f"{model}_full_risk"] = (score_pair, [True, True])
            score_columns.extend(score_pair)

    departure = rows["current_lots_available"]
    arrival = rows["actual_lots_available"]
    rows["_observed"] = (
        rows["current_is_valid"].eq(True).fillna(False)
        & rows["actual_is_observed"].eq(True).fillna(False)
        & np.isfinite(departure) & departure.ge(0)
        & np.isfinite(arrival) & arrival.ge(0)
    ).fillna(False)
    scores = rows[["current_occupancy", *score_columns]]
    rows["_complete"] = (rows["_observed"] & scores.notna().all(axis=1)
                         & scores.ge(0).all(axis=1) & scores.le(1).all(axis=1))
    transitions, results = [], []
    for horizon, at_horizon in rows.groupby("horizon_min", sort=True):
        observed = at_horizon.loc[at_horizon["_observed"]]
        departure, arrival = observed["current_lots_available"], observed["actual_lots_available"]
        open_departure, safe_departure = departure.gt(0), departure.ge(threshold)
        n = len(observed)
        declined = int(arrival.lt(departure).sum())
        open_count, safe_count = int(open_departure.sum()), int(safe_departure.sum())
        open_to_full = int((open_departure & arrival.eq(0)).sum())
        safe_to_low = int((safe_departure & arrival.lt(threshold)).sum())
        transitions.append({
            "horizon_min": horizon, "matched_rows": n,
            "vacancy_declined_count": declined, "vacancy_declined_rate": _rate(declined, n),
            "open_at_departure_count": open_count, "open_to_full_count": open_to_full,
            "open_to_full_rate": _rate(open_to_full, open_count), "safe_at_departure_count": safe_count,
            "safe_to_low_count": safe_to_low, "safe_to_low_rate": _rate(safe_to_low, safe_count),
            "low_spaces_threshold": threshold,
        })

        groups = at_horizon.groupby("decision_time", sort=False)["_complete"]
        candidates = at_horizon.loc[
            groups.transform("size").eq(len(parks)) & groups.transform("all")
            & at_horizon["current_lots_available"].gt(0)
        ]
        origins = int(candidates["decision_time"].nunique())
        for name, (columns, ascending) in strategies.items():
            ordered = candidates.sort_values(
                ["decision_time", *columns, "carpark_id"], ascending=[True, *ascending, True]
            )
            ranks = ordered.groupby("decision_time", sort=False)
            first = ranks.head(1)["actual_lots_available"]
            top_three = ranks.head(3)
            top_three_available = int(top_three.loc[top_three["actual_lots_available"].gt(0),
                                                    "decision_time"].nunique())
            first_full = int(first.eq(0).sum())
            first_low = int(first.lt(threshold).sum())
            results.append({
                "horizon_min": horizon, "strategy": name, "matched_origins": origins,
                "first_full_count": first_full, "first_full_rate": _rate(first_full, origins),
                "first_low_spaces_count": first_low, "first_low_spaces_rate": _rate(first_low, origins),
                "top_three_any_available_count": top_three_available,
                "top_three_any_available_rate": _rate(top_three_available, origins),
                "low_spaces_threshold": threshold,
            })
    return pd.DataFrame(transitions, columns=TRANSITION_COLUMNS), pd.DataFrame(results, columns=REPLAY_COLUMNS)
