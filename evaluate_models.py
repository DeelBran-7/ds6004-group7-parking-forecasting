"""Part C held-out model scores on shared, observed carpark-time pairs."""

import numpy as np
import pandas as pd

from evaluation_stats import paired_day_inference
from risk_calibration import reliability_bins, risk_metrics


OCCUPANCY_MODELS = ("B0", "B1", "M0", "M1")
RISK_MODELS = ("persistence", "occupancy_proxy", "xgboost", "catboost")
OCCUPANCY_PAIRS = (("M0", "B0"), ("M1", "B0"), ("M1", "M0"))
RISK_PAIRS = (("xgboost", "occupancy_proxy"), ("catboost", "occupancy_proxy"),
              ("xgboost", "catboost"))
KEY = ["decision_time", "horizon_min", "carpark_id"]
BASE_COLUMNS = [*KEY, "current_is_valid", "actual_is_observed", "actual_lots_available",
                "actual_total_lots"]


def _test_frame(frame, required, *, require_split):
    """Validate held-out keys and observed counts without trusting derived targets."""
    required = {*BASE_COLUMNS, *required}
    if require_split:
        required.add("split")
    if missing := required.difference(frame.columns):
        raise ValueError(f"Missing model evaluation columns: {', '.join(sorted(missing))}")
    frame = (frame.loc[frame["split"].eq("test")] if "split" in frame else frame).reset_index(drop=True)
    if frame.empty:
        raise ValueError("No held-out model evaluation rows")
    frame["decision_time"] = pd.to_datetime(frame["decision_time"], format="ISO8601")
    if frame["decision_time"].dt.tz is not None:
        raise ValueError("Model evaluation expects naive Singapore local timestamps")
    if frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
        raise ValueError("Model evaluation requires unique, nonmissing decision/horizon/carpark keys")
    horizons = sorted(frame["horizon_min"].unique())
    valid = frame["current_is_valid"].eq(True) & frame["actual_is_observed"].eq(True)
    if "forecast_label_eligible" in frame:
        valid &= frame["forecast_label_eligible"].eq(True)
    frame = frame.loc[valid].copy()
    free, cap = frame["actual_lots_available"], frame["actual_total_lots"]
    if not (np.isfinite(free) & np.isfinite(cap) & cap.gt(0) & free.ge(0) & free.le(cap)).all():
        raise ValueError("Valid held-out labels require observed arrival counts in [0, capacity]")
    frame["day"] = frame["decision_time"].dt.normalize()
    return frame, horizons


def _shared_predictions(frame, columns):
    """Use all available models at the horizon; omit a wholly absent forecast."""
    predictions = frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
    predictions = predictions.where(predictions.ge(0) & predictions.le(1))
    active = predictions.columns[predictions.notna().any()].tolist()
    omitted = [model for model in columns if model not in active]
    common = predictions[active].notna().all(axis=1) if active else pd.Series(False, index=frame.index)
    return predictions, active, omitted, common


def _coverage(predictions, model, active, omitted, cohort, common):
    n_eligible = int(cohort.sum())
    n_available = int((predictions[model].notna() & cohort).sum())
    n_scored = int((common & cohort).sum()) if model in active else 0
    status = ("empty_cohort" if not n_eligible else "model_unavailable" if model not in active
              else "no_shared_predictions" if not n_scored else "scored")
    return {"n_eligible": n_eligible, "n_available": n_available, "n_scored": n_scored,
            "omitted_models": ";".join(omitted), "status": status}


def _pair_rows(day, losses, pairs, key, metric, repetitions, seed):
    for candidate, reference in pairs:
        # An unavailable baseline has no pairs and no effect estimate.
        if candidate in losses and reference in losses:
            statistic = paired_day_inference(day, losses[candidate], losses[reference],
                                             repetitions=repetitions, seed=seed)
        else:
            statistic = paired_day_inference([], [], [], repetitions=repetitions, seed=seed)
        yield dict(key, comparison=f"{candidate}-{reference}", candidate=candidate,
                   reference=reference, metric=metric, **statistic)


def evaluate_forecasts(frame, *, repetitions, seed, include_comparisons=True):
    """Evaluate B0/B1/M0/M1 on shared test rows; error units are occupancy ratios."""
    data, horizons = _test_frame(frame, ["actual_occupancy", "is_holiday"], require_split=True)
    actual = 1.0 - data["actual_lots_available"] / data["actual_total_lots"]
    if not np.allclose(actual, data["actual_occupancy"], rtol=0, atol=1e-9):
        raise ValueError("Observed occupancy does not match arrival counts and capacity")
    near = data["actual_lots_available"].mul(10).lt(data["actual_total_lots"])
    metrics, comparisons = [], []
    for horizon in horizons:
        sample = data.loc[data["horizon_min"].eq(horizon)]
        labels, near_labels = actual.loc[sample.index], near.loc[sample.index]
        predictions, active, omitted, common = _shared_predictions(sample, OCCUPANCY_MODELS)
        cohorts = {"all": pd.Series(True, index=sample.index), "near_full": near_labels,
                   "holiday": sample["is_holiday"].eq(1), "non_holiday": sample["is_holiday"].eq(0)}
        for cohort, condition in cohorts.items():
            keep = common & condition
            key = {"horizon_min": horizon, "cohort": cohort}
            losses = predictions.loc[keep, active].sub(labels.loc[keep], axis=0).abs()
            if include_comparisons:
                comparisons.extend(_pair_rows(sample.loc[keep, "day"], losses, OCCUPANCY_PAIRS,
                                              key, "mae", repetitions, seed))
            for model in OCCUPANCY_MODELS:
                count = _coverage(predictions, model, active, omitted, condition, common)
                pred = predictions.loc[keep, model] if model in active else predictions.loc[[], model]
                error = losses[model] if model in active else pred
                observed_near = near_labels.loc[pred.index]
                predicted_near = pred.gt(0.9)
                tp = int((observed_near & predicted_near).sum())
                fp = int((~observed_near & predicted_near).sum())
                fn = int((observed_near & ~predicted_near).sum())
                metrics.append(dict(
                    key, model=model, **count,
                    mae=float(error.mean()) if len(error) else None,
                    rmse=float(np.sqrt(error.pow(2).mean())) if len(error) else None,
                    near_full_tp=tp, near_full_fp=fp, near_full_fn=fn,
                    # A positives-only slice cannot estimate classifier precision.
                    near_full_precision=tp / (tp + fp) if tp + fp and cohort != "near_full" else None,
                    near_full_recall=tp / (tp + fn) if tp + fn else None,
                ))
    return pd.DataFrame(metrics), pd.DataFrame(comparisons)


def evaluate_risk(frame, *, repetitions, seed):
    """Score final probabilities and reliability bins against exact arrival events."""
    data, horizons = _test_frame(frame, ["current_lots_available", "low_spaces_threshold"],
                                 require_split=False)
    thresholds = pd.to_numeric(frame["low_spaces_threshold"], errors="coerce").unique()
    if len(thresholds) != 1 or not np.isfinite(thresholds[0]) or thresholds[0] < 1 or thresholds[0] % 1:
        raise ValueError("Risk forecasts require a single positive integer low-space threshold")
    threshold = int(thresholds[0])
    if not (np.isfinite(data["current_lots_available"]) & data["current_lots_available"].ge(0)).all():
        raise ValueError("Valid risk rows require nonnegative observed departure availability")
    targets = {"full": data["actual_lots_available"].eq(0),
               "low_spaces": data["actual_lots_available"].lt(threshold)}
    metrics, reliability, comparisons = [], [], []
    for event, observed in targets.items():
        supplied_label = f"actual_{event}"
        if supplied_label in data and not data[supplied_label].eq(observed.astype(int)).all():
            raise ValueError(f"{supplied_label} does not match observed arrival counts")
        columns = {f"p_{event}_{model}": model for model in RISK_MODELS}
        for horizon in horizons:
            sample = data.loc[data["horizon_min"].eq(horizon)]
            probs = sample.reindex(columns=columns).rename(columns=columns)
            predictions, active, omitted, common = _shared_predictions(probs, RISK_MODELS)
            cohorts = {"all": pd.Series(True, index=sample.index),
                       "open_at_departure": sample["current_lots_available"].gt(0)}
            for cohort, condition in cohorts.items():
                keep = common & condition
                key = {"horizon_min": horizon, "event": event, "cohort": cohort}
                truth = observed.loc[sample.index].loc[keep].astype(int)
                losses = predictions.loc[keep, active].sub(truth, axis=0).pow(2)
                comparisons.extend(_pair_rows(sample.loc[keep, "day"], losses, RISK_PAIRS,
                                              key, "brier", repetitions, seed))
                for model in RISK_MODELS:
                    count = _coverage(predictions, model, active, omitted, condition, common)
                    pred = predictions.loc[keep, model] if model in active else predictions.loc[[], model]
                    y = truth.loc[pred.index].to_numpy(dtype=float)
                    p = pred.to_numpy(dtype=float)
                    scores = risk_metrics(y, p)
                    scores.pop("n")
                    metrics.append(dict(key, model=model, **count, low_spaces_threshold=threshold, **scores))
                    reliability.extend(dict(key, model=model, **row) for row in reliability_bins(y, p))
    return pd.DataFrame(metrics), pd.DataFrame(reliability), pd.DataFrame(comparisons)
