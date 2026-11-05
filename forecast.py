"""DS6004 Group 7: occupancy forecasting for 15- and 30-minute horizons."""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


HORIZONS = (15, 30)
EMBARGO = pd.Timedelta(minutes=max(HORIZONS))
LAGS = (15, 30, 45, 60)
REQUIRED = {"timestamp", "carpark_id", "total_lots", "lots_available", "lag_seconds"}
FEATURES = [
    "current_occupancy", "total_lots", "lag_seconds", "target_slot_sin",
    "target_slot_cos", "target_weekday", "target_is_weekend",
    *(f"occ_lag_{lag}" for lag in LAGS),
    "occ_roll_mean_60", "occ_roll_std_60", "occ_roll_count_60",
]
OUTPUT_COLUMNS = [
    "decision_time", "target_time", "carpark_id", "horizon_min", "split",
    "current_occupancy", "total_lots", "current_lots_available", "lag_seconds",
    "current_is_valid", "is_holiday", "actual_is_observed", "actual_occupancy",
    "actual_total_lots", "actual_lots_available", "B0", "B1", "M0", "M1",
]


def load_history(path, max_lag_seconds=900, carparks=None):
    """Load the pilot schema; keep invalid snapshots flagged but unusable."""
    if not math.isfinite(max_lag_seconds) or max_lag_seconds < 0:
        raise ValueError("max_lag_seconds must be finite and nonnegative")
    frame = pd.read_csv(path, dtype={"carpark_id": "string"})
    if missing := REQUIRED.difference(frame.columns):
        raise ValueError(f"Missing history columns: {', '.join(sorted(missing))}")
    if "lot_type" in frame:
        frame = frame.loc[frame["lot_type"].astype("string").str.strip().str.upper().eq("C")].copy()
    frame["carpark_id"] = frame["carpark_id"].str.strip()
    if frame["carpark_id"].isna().any() or frame["carpark_id"].eq("").any():
        raise ValueError("History contains a missing carpark_id")
    if carparks:
        if missing := set(carparks).difference(frame["carpark_id"]):
            raise ValueError(f"Requested carparks not found: {', '.join(sorted(missing))}")
        frame = frame.loc[frame["carpark_id"].isin(carparks)].copy()
    if frame.empty:
        raise ValueError("No carpark observations to process")

    # Naive inputs already use Singapore time; explicit offsets are converted.
    raw = frame["timestamp"].astype("string").str.strip()
    aware = raw.str.contains(r"(?:[Zz]|[+-]\d{2}:?\d{2})$", na=False)
    times = pd.to_datetime(raw.mask(aware), format="ISO8601", errors="coerce")
    if aware.any():
        times.loc[aware] = (pd.to_datetime(raw.loc[aware], format="ISO8601", errors="coerce", utc=True)
                            .dt.tz_convert("Asia/Singapore").dt.tz_localize(None))
    if times.isna().any() or times.ne(times.dt.floor("15min")).any():
        raise ValueError("Every timestamp must be valid and on the 15-minute grid")
    frame["timestamp"] = times
    if frame.duplicated(["timestamp", "carpark_id"]).any():
        raise ValueError("Duplicate (timestamp, carpark_id) keys; resolve duplicates before forecasting")

    for col in ("total_lots", "lots_available", "lag_seconds"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(float)
    capacity, free, age = (frame[c] for c in ("total_lots", "lots_available", "lag_seconds"))
    with np.errstate(invalid="ignore"):
        bad_counts = (~np.isfinite(capacity) | ~np.isfinite(free) | (capacity <= 0)
                      | (free < 0) | (free > capacity) | (capacity % 1 != 0) | (free % 1 != 0))
    bad_age = ~np.isfinite(age) | (age < 0)
    stale = age > max_lag_seconds
    valid = ~(bad_counts | bad_age | stale)
    frame["is_valid"] = valid
    frame["occupancy"] = 1.0 - free.where(valid) / capacity.where(valid)
    observed_slots = int(times.nunique())
    grid_slots = int((times.max() - times.min()) // pd.Timedelta(minutes=15)) + 1
    counts = frame.groupby("carpark_id", observed=True).size()
    audit = {
        "timezone": "Asia/Singapore", "first_time": times.min().isoformat(),
        "last_time": times.max().isoformat(), "rows": len(frame),
        "rows_per_carpark": {key: int(value) for key, value in counts.items()},
        "observed_slots": observed_slots, "grid_slots": grid_slots,
        "missing_carpark_slots": grid_slots * len(counts) - len(frame),
        "invalid_count_rows": int(bad_counts.sum()), "invalid_lag_rows": int(bad_age.sum()),
        "stale_rows": int(stale.sum()), "valid_rows": int(valid.sum()),
        "full_rows": int((valid & free.eq(0)).sum()),
        "near_full_rows_including_full": int((valid & free.mul(10).lt(capacity)).sum()),
        "max_lag_seconds": max_lag_seconds,
    }
    return frame.sort_values(["timestamp", "carpark_id"]).reset_index(drop=True), audit


@dataclass(frozen=True)
class HolidayCalendar:
    dates: frozenset
    covered_years: frozenset

    def flags(self, timestamps):
        flags = timestamps.dt.normalize().isin(self.dates).astype(float)
        # Missing calendar years are unknown, not ordinary days.
        return flags.where(timestamps.dt.year.isin(self.covered_years))


def load_holidays(path):
    """Read complete calendar years from MOM's consolidated holidays CSV."""
    calendar = pd.read_csv(path)
    if "date" not in calendar:
        raise ValueError("Holiday CSV requires a date column")
    dates = pd.to_datetime(calendar["date"], format="%Y-%m-%d", errors="coerce")
    if dates.empty or dates.isna().any():
        raise ValueError("Holiday CSV requires valid YYYY-MM-DD dates")
    return HolidayCalendar(frozenset(dates), frozenset(dates.dt.year))


def split_times(decision_time):
    """Split on observed decision timestamps, embargoing preceding origins."""
    times = pd.DatetimeIndex(decision_time.unique()).sort_values()
    if len(times) < 5:
        raise ValueError("At least five distinct timestamps are required for a 60/20/20 split")
    validation_start, test_start = times[len(times) * 3 // 5], times[len(times) * 4 // 5]
    split = np.select([decision_time < validation_start, decision_time < test_start],
                      ["train", "validation"], default="test")
    for boundary in (validation_start, test_start):
        mask = (decision_time >= boundary - EMBARGO) & (decision_time < boundary)
        split[mask] = "embargo"
    return split


def make_examples(history, holidays, horizon_min):
    """Build features and labels by exact carpark and time, even across gaps."""
    if horizon_min not in HORIZONS:
        raise ValueError(f"Unsupported horizon: {horizon_min}")
    horizon = pd.Timedelta(minutes=horizon_min)
    index = history.set_index(["carpark_id", "timestamp"])
    observed = index[["occupancy", "total_lots", "lots_available"]].where(index["is_valid"], axis=0)
    examples = history[["timestamp", "carpark_id", "occupancy", "total_lots", "lots_available",
                        "lag_seconds", "is_valid"]].rename(columns={
        "timestamp": "decision_time", "occupancy": "current_occupancy",
        "lots_available": "current_lots_available", "is_valid": "current_is_valid",
    }).copy()
    origin = examples["decision_time"]
    target = examples["target_time"] = origin + horizon

    def exact(times, columns):
        keys = pd.MultiIndex.from_arrays([examples["carpark_id"], times], names=index.index.names)
        return observed[columns].reindex(keys).to_numpy()

    examples["horizon_min"] = horizon_min
    examples["split"] = split_times(origin)
    examples["B0"] = examples["current_occupancy"]
    examples["B1"] = exact(target - pd.Timedelta(days=7), "occupancy")
    for lag in LAGS:
        examples[f"occ_lag_{lag}"] = exact(origin - pd.Timedelta(minutes=lag), "occupancy")
    # One-hour window (t-45, t-30, t-15, t), allowing missing history.
    window = examples[["current_occupancy", "occ_lag_15", "occ_lag_30", "occ_lag_45"]]
    examples["occ_roll_mean_60"] = window.mean(axis=1)
    examples["occ_roll_std_60"] = window.std(axis=1, ddof=0)
    examples["occ_roll_count_60"] = window.count(axis=1)
    phase = 2 * np.pi * (target.dt.hour * 60 + target.dt.minute) / 1440
    examples["target_slot_sin"], examples["target_slot_cos"] = np.sin(phase), np.cos(phase)
    examples["target_weekday"] = target.dt.dayofweek
    examples["target_is_weekend"] = target.dt.dayofweek.ge(5).astype(int)
    examples["is_holiday"] = holidays.flags(target)
    labels = ["actual_occupancy", "actual_total_lots", "actual_lots_available"]
    examples[labels] = exact(target, ["occupancy", "total_lots", "lots_available"])
    examples["actual_is_observed"] = examples["actual_occupancy"].notna()
    return examples


def fit_predict(examples):
    """Train paired XGBoost regressors, predicting validation and test origins."""
    import xgboost as xgb

    train = (examples["split"].eq("train") & examples["current_is_valid"]
             & examples["actual_is_observed"])
    predict = examples["split"].isin(["validation", "test"]) & examples["current_is_valid"]
    if not train.any():
        raise ValueError("No usable training labels after the embargo and quality checks")
    matrix = examples[FEATURES].copy()
    # Establish park indicators using training rows only; unseen parks are all-zero.
    carparks = sorted(examples.loc[train, "carpark_id"].unique())
    for i, carpark in enumerate(carparks):
        matrix[f"park_{i}"] = examples["carpark_id"].eq(carpark).astype(int)
    models = {}
    params = {"objective": "reg:squarederror", "max_depth": 4, "eta": 0.05,
              "tree_method": "hist", "seed": 42, "nthread": 2}
    for name in ("M0", "M1"):
        features = matrix if name == "M0" else matrix.assign(is_holiday=examples["is_holiday"])
        training = xgb.DMatrix(features.loc[train], label=examples.loc[train, "actual_occupancy"], nthread=2)
        model = xgb.train(params, training, num_boost_round=200)
        model.set_attr(carpark_indicator_ids=json.dumps(carparks))
        examples[name] = np.nan
        if predict.any():
            predictions = model.predict(xgb.DMatrix(features.loc[predict], nthread=2))
            examples.loc[predict, name] = predictions.clip(0, 1)
        models[name] = model
    return models


def pilot_metrics(examples):
    """Smoke-test metrics with each baseline's actual denominator."""
    test = examples.loc[examples["split"].eq("test") & examples["current_is_valid"]
                        & examples["actual_is_observed"]]
    groups = {
        "all": test,
        "near_full": test.loc[test["actual_lots_available"].mul(10).lt(test["actual_total_lots"])],
        "holiday": test.loc[test["is_holiday"].eq(1)],
        "non_holiday": test.loc[test["is_holiday"].eq(0)],
    }
    for group, frame in groups.items():
        for model in ("B0", "B1", "M0", "M1"):
            scored = frame.loc[frame[model].notna()]
            error = scored[model] - scored["actual_occupancy"]
            observed_risk = scored["actual_lots_available"].mul(10).lt(scored["actual_total_lots"])
            predicted_risk = scored[model].gt(0.9)
            true_positives = int((observed_risk & predicted_risk).sum())
            observed_risks = int(observed_risk.sum())
            predicted_risks = int(predicted_risk.sum())
            yield {
                "horizon_min": int(examples["horizon_min"].iat[0]), "group": group,
                "model": model, "eligible_labels": len(frame), "n_scored": len(error),
                "mae": float(error.abs().mean()) if len(error) else None,
                "rmse": float(np.sqrt(error.pow(2).mean())) if len(error) else None,
                "near_full_precision": true_positives / predicted_risks if predicted_risks else None,
                "near_full_recall": true_positives / observed_risks if observed_risks else None,
            }


def run(history_path, holidays_path, output_dir, max_lag_seconds=900, carparks=None):
    history, audit = load_history(history_path, max_lag_seconds, carparks)
    calendar = load_holidays(holidays_path)
    output_dir = Path(output_dir)
    forecasts, metrics, models = [], [], {}
    audit["horizons"] = {}
    for horizon in HORIZONS:
        examples = make_examples(history, calendar, horizon)
        models[horizon] = fit_predict(examples)
        metrics.extend(pilot_metrics(examples))
        forecasts.append(examples.loc[examples["split"].isin(["validation", "test"]), OUTPUT_COLUMNS])
        audit["horizons"][str(horizon)] = {}
        for split in ("train", "validation", "test", "embargo"):
            part = examples.loc[examples["split"].eq(split)]
            labeled = part.loc[part["current_is_valid"] & part["actual_is_observed"]]
            audit["horizons"][str(horizon)][split] = {
                "rows": len(part), "observed_labels_with_valid_origin": len(labeled),
                "first_decision_time": part["decision_time"].min().isoformat() if len(part) else None,
                "last_decision_time": part["decision_time"].max().isoformat() if len(part) else None,
                "holiday_days": int(labeled.loc[labeled["is_holiday"].eq(1), "target_time"].dt.date.nunique()),
                "unknown_holiday_labels": int(labeled["is_holiday"].isna().sum()),
                "b1_matches": int(labeled["B1"].notna().sum()),
            }
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(forecasts, ignore_index=True).to_csv(
        output_dir / "predictions.csv", index=False, date_format="%Y-%m-%dT%H:%M:%S"
    )
    pd.DataFrame(metrics).to_csv(output_dir / "forecast_diagnostics.csv", index=False)
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    for horizon, fitted in models.items():
        for name, model in fitted.items():
            model.save_model(output_dir / f"{name}_{horizon}min.json")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--holidays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-lag-seconds", type=float, default=900)
    parser.add_argument("--carparks", nargs="+", help="Optional list of pilot carpark IDs")
    args = parser.parse_args()
    try:
        audit = run(args.history, args.holidays, args.output, args.max_lag_seconds, args.carparks)
    except ValueError as error:
        parser.error(str(error))
    print(f"Processed {audit['rows']} snapshots across {len(audit['rows_per_carpark'])} carparks.")
    print(f"Wrote forecasts and diagnostics to {args.output}")


if __name__ == "__main__":
    main()
