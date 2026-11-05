"""Forecast absolute parking-space risks, then calibrate on later observations."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecast import FEATURES, HORIZONS, load_history, load_holidays, make_examples
from risk_calibration import fit_calibrator, reliability_bins, risk_metrics
from risk_replay import summarize_replay


EVENTS = ("full", "low_spaces")
MODELS = ("persistence", "occupancy_proxy", "xgboost", "catboost")
RISK_FEATURES = [*FEATURES, "current_lots_available", "occ_change_15", "is_holiday"]
OUTPUT_BASE = [
    "decision_time", "target_time", "carpark_id", "horizon_min",
    "current_is_valid", "current_lots_available", "current_occupancy", "total_lots",
    "actual_is_observed", "actual_lots_available", "actual_total_lots",
    "actual_full", "actual_low_spaces", "low_spaces_threshold",
]
XGB_PARAMS = {"max_depth": 4, "eta": 0.05, "tree_method": "hist", "seed": 42, "nthread": 2}
ROUNDS = 200


def add_risk_targets(examples, threshold):
    """Use actual arrival counts; preserve unknown labels as missing."""
    if threshold < 1:
        raise ValueError("The low-space threshold must be a positive integer")
    observed = examples["actual_is_observed"]
    free = examples["actual_lots_available"]
    examples["actual_full"] = free.eq(0).astype(float).where(observed)
    examples["actual_low_spaces"] = free.lt(threshold).astype(float).where(observed)
    examples["low_spaces_threshold"] = threshold
    examples["occ_change_15"] = examples["current_occupancy"] - examples["occ_lag_15"]


def reconcile_nested(full, low):
    """Keep the primary full-risk estimate and raise inconsistent low risk."""
    full, low = np.asarray(full, dtype=float).copy(), np.asarray(low, dtype=float).copy()
    conflict = full > low
    low[conflict] = full[conflict]
    return full, low, int(conflict.sum())


def feature_matrix(examples, train):
    features = examples[RISK_FEATURES].copy()
    for i, park in enumerate(sorted(examples.loc[train, "carpark_id"].unique())):
        features[f"park_{i}"] = examples["carpark_id"].eq(park).astype(int)
    # Constant columns have no predictive content and can prevent CatBoost fitting.
    varying = features.loc[train].nunique(dropna=False).gt(1)
    return features.loc[:, varying]


def raw_scores(examples, train, score, threshold):
    """Use the same train-only features for boosted risk and occupancy models."""
    import xgboost as xgb
    from catboost import CatBoostClassifier

    features = feature_matrix(examples, train)
    if features.shape[1] == 0:
        raise ValueError("No varying training features for risk forecasting")
    train_features, score_features = features.loc[train], features.loc[score]
    score_matrix = xgb.DMatrix(score_features, nthread=2)

    def xgboost_score(target, objective):
        training = xgb.DMatrix(train_features, label=target, nthread=2)
        model = xgb.train(dict(XGB_PARAMS, objective=objective), training, num_boost_round=ROUNDS)
        return model.predict(score_matrix).clip(0, 1)

    occupancy = xgboost_score(examples.loc[train, "actual_occupancy"], "reg:squarederror")
    # Capacity at decision time is known; arrival capacity is never a model feature.
    expected_free = (1.0 - occupancy.astype(float)) * examples.loc[score, "total_lots"].to_numpy()
    occupancy_score = 1.0 / (1.0 + expected_free)  # A score until fitted to each event.
    now_free = examples.loc[score, "current_lots_available"].to_numpy()

    for event in EVENTS:
        labels = examples.loc[train, f"actual_{event}"]
        persistence = now_free == 0 if event == "full" else now_free < threshold
        if labels.nunique() < 2:
            # Training data have no evidence for learning a decision boundary.
            prior = np.full(len(score_features), (labels.sum() + 1) / (len(labels) + 2))
            yield event, {"persistence": persistence.astype(float), "occupancy_proxy": occupancy_score,
                          "xgboost": prior, "catboost": prior}, "single_class_training_prior"
            continue
        catboost = CatBoostClassifier(
            iterations=ROUNDS, depth=4, learning_rate=0.05, loss_function="Logloss",
            random_seed=42, thread_count=2, verbose=False, allow_writing_files=False,
        )
        catboost.fit(train_features, labels.astype(int))
        yield event, {
            "persistence": persistence.astype(float), "occupancy_proxy": occupancy_score,
            "xgboost": xgboost_score(labels, "binary:logistic"),
            "catboost": catboost.predict_proba(score_features)[:, 1],
        }, "fitted"


def forecast_horizon(history, holidays, horizon, threshold, min_calibration_class_count):
    examples = make_examples(history, holidays, horizon)
    add_risk_targets(examples, threshold)
    usable = examples["current_is_valid"] & examples["actual_is_observed"]
    train = examples["split"].eq("train") & usable
    calibration = examples["split"].eq("validation") & usable
    test = examples["split"].eq("test")
    score = examples["split"].isin(["validation", "test"]) & examples["current_is_valid"]
    if not train.any() or not (test & usable).any():
        raise ValueError("Risk fitting and evaluation require usable train and test labels after embargo")
    training_end = examples.loc[train, "target_time"].max()
    test_start = examples.loc[test, "decision_time"].min()
    if calibration.any():
        assert training_end < examples.loc[calibration, "decision_time"].min()
        assert examples.loc[calibration, "target_time"].max() < test_start

    diagnostics = []
    score_index = examples.index[score]
    cal_in_scores = calibration.loc[score_index].to_numpy()
    for event, scores, train_status in raw_scores(examples, train, score, threshold):
        labels = examples.loc[calibration, f"actual_{event}"].to_numpy()
        for model, raw in scores.items():
            calibrator = fit_calibrator(raw[cal_in_scores], labels, min_per_class=min_calibration_class_count)
            raw_col, prob_col = f"raw_score_{event}_{model}", f"p_{event}_{model}"
            examples[raw_col] = np.nan
            examples.loc[score, raw_col] = raw
            examples[prob_col] = np.nan
            # An uncalibrated occupancy score has no probability interpretation.
            if model != "occupancy_proxy" or calibrator.status == "platt":
                examples.loc[score, prob_col] = calibrator.predict(raw)
            diagnostics.append({
                "horizon_min": horizon, "event": event, "model": model,
                "train_rows": int(train.sum()), "train_events": int(examples.loc[train, f"actual_{event}"].sum()),
                "training_status": train_status if model in ("xgboost", "catboost") else "baseline",
                "calibration_rows": len(labels), "calibration_events": int(labels.sum()),
                "calibration_status": calibrator.status, "slope": calibrator.slope,
                "intercept": calibrator.intercept,
                "calibration_first_decision": examples.loc[calibration, "decision_time"].min(),
                "calibration_last_target": examples.loc[calibration, "target_time"].max(),
                "train_last_target": training_end, "test_first_decision": test_start,
            })

    adjustments = {}
    for model in MODELS:
        full, low = f"p_full_{model}", f"p_low_spaces_{model}"
        examples.loc[test, full], examples.loc[test, low], adjustments[model] = reconcile_nested(
            examples.loc[test, full], examples.loc[test, low]
        )
    prob_cols = [f"p_{event}_{model}" for event in EVENTS for model in MODELS]
    raw_cols = [f"raw_score_{event}_{model}" for event in EVENTS for model in MODELS]
    return examples.loc[test, [*OUTPUT_BASE, *prob_cols, *raw_cols]], diagnostics, adjustments


def test_diagnostics(predictions, threshold):
    """Score test rows only; retain denominators for at-risk subsets."""
    valid = predictions["current_is_valid"] & predictions["actual_is_observed"]
    groups = {"all": valid, "open_at_departure": valid & predictions["current_lots_available"].gt(0),
              "safe_at_departure": valid & predictions["current_lots_available"].ge(threshold)}
    metrics, reliability = [], []
    for horizon in HORIZONS:
        for event in EVENTS:
            label = f"actual_{event}"
            for model in MODELS:
                variants = {"final": f"p_{event}_{model}"}
                if model != "occupancy_proxy":
                    variants["raw"] = f"raw_score_{event}_{model}"
                for version, probability in variants.items():
                    for name, mask in groups.items():
                        cohort = predictions.loc[mask & predictions["horizon_min"].eq(horizon)]
                        rows = cohort.loc[cohort[probability].notna()]
                        y, p = rows[label].to_numpy(), rows[probability].to_numpy()
                        key = {"horizon_min": horizon, "event": event, "model": model,
                               "version": version, "cohort": name}
                        metrics.append(dict(key, eligible_rows=len(cohort), **risk_metrics(y, p)))
                        if version == "final" and name != "safe_at_departure":
                            reliability.extend(dict(key, **item) for item in reliability_bins(y, p))
    return pd.DataFrame(metrics), pd.DataFrame(reliability)


def run(history_path, holidays_path, output_dir, threshold=3, max_lag_seconds=900,
        carparks=None, min_calibration_class_count=5):
    if any(not isinstance(value, (int, np.integer)) or value < 1
           for value in (threshold, min_calibration_class_count)):
        raise ValueError("Threshold and minimum calibration class count must be positive integers")
    history, audit = load_history(history_path, max_lag_seconds, carparks)
    holidays = load_holidays(holidays_path)
    predictions, calibration, adjustments = [], [], {}
    transition_tables, replay_tables = [], []
    omitted_replay_models = {}
    for horizon in HORIZONS:
        rows, details, projected = forecast_horizon(history, holidays, horizon, threshold,
                                                     min_calibration_class_count)
        predictions.append(rows)
        calibration.extend(details)
        adjustments[str(horizon)] = projected
        # A model with no score pair at all (e.g. uncalibrated occupancy proxy)
        # cannot enter this horizon's replay. Partial gaps still exclude cases
        # for every included strategy, keeping comparisons on matched origins.
        pairs = {model: [f"p_{event}_{model}" for event in EVENTS] for model in MODELS}
        omitted = [model for model, columns in pairs.items() if not rows[columns].notna().all(axis=1).any()]
        omitted_replay_models[str(horizon)] = omitted
        replay_input = rows.drop(columns=[col for model in omitted for col in pairs[model]])
        transitions, replay = summarize_replay(replay_input, list(audit["rows_per_carpark"]))
        transition_tables.append(transitions)
        replay_tables.append(replay)
    predictions = pd.concat(predictions, ignore_index=True)
    metrics, reliability = test_diagnostics(predictions, threshold)
    audit.update(low_spaces_threshold=threshold, min_calibration_class_count=min_calibration_class_count,
                 adjusted_test_probability_pairs=adjustments, omitted_replay_models=omitted_replay_models)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    files = {"risk_predictions": predictions, "risk_metrics": metrics, "risk_reliability": reliability,
             "risk_calibration": pd.DataFrame(calibration),
             "availability_transitions": pd.concat(transition_tables, ignore_index=True),
             "risk_replay": pd.concat(replay_tables, ignore_index=True)}
    for name, frame in files.items():
        frame.to_csv(destination / f"{name}.csv", index=False, date_format="%Y-%m-%dT%H:%M:%S")
    (destination / "risk_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--holidays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--low-spaces-threshold", type=int, default=3)
    parser.add_argument("--max-lag-seconds", type=float, default=900)
    parser.add_argument("--carparks", nargs="+")
    parser.add_argument("--min-calibration-class-count", type=int, default=5)
    args = parser.parse_args()
    try:
        run(args.history, args.holidays, args.output, args.low_spaces_threshold,
            args.max_lag_seconds, args.carparks, args.min_calibration_class_count)
    except ValueError as error:
        parser.error(str(error))
    print(f"Wrote 15- and 30-minute risk forecasts and test diagnostics to {args.output}")


if __name__ == "__main__":
    main()
