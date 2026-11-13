"""Summarize existing held-out full-risk predictions by car park.

This is a read-only analysis of saved predictions, not model training. It uses
the same observation and common-probability filters as evaluate_models.py,
restricts the cohort to spaces available at departure, and checks the pooled
counts and scores against the published evaluation before saving a small CSV.
AP is undefined when a park has only one observed class, and is left blank.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_models import RISK_MODELS, _shared_predictions, _test_frame
from risk_calibration import risk_metrics


REPORT_MODELS = ("occupancy_proxy", "xgboost", "catboost")
DEFAULT_STUDY = ROOT / "outputs/history_comparison/two_year_standard"
DEFAULT_REFERENCE = ROOT / "examples/history_comparison/two_year_standard/evaluation/risk_probability_metrics.csv"
DEFAULT_OUTPUT = ROOT / "reports/model_section/tables/risk_by_carpark.csv"


def summarize(predictions_path, reference_path):
    probability_columns = {f"p_full_{model}": model for model in RISK_MODELS}
    columns = {
        "decision_time", "carpark_id", "horizon_min", "current_is_valid",
        "actual_is_observed", "forecast_label_eligible", "current_lots_available",
        "actual_lots_available", "actual_total_lots", *probability_columns,
    }
    predictions = pd.read_csv(predictions_path, usecols=sorted(columns),
                              dtype={"carpark_id": "string"})
    # Require the strict label flag rather than silently accepting an older file
    # with only valid endpoints. This matters at the 30-minute horizon.
    data, horizons = _test_frame(
        predictions, ["current_lots_available", "forecast_label_eligible"],
        require_split=False,
    )
    if horizons != [15, 30]:
        raise ValueError("Expected the saved 15- and 30-minute test predictions")
    reference = pd.read_csv(reference_path)
    result = []
    for horizon in horizons:
        sample = data.loc[data.horizon_min.eq(horizon)].copy()
        probs = sample[list(probability_columns)].rename(columns=probability_columns)
        values, active, omitted, common = _shared_predictions(probs, RISK_MODELS)
        if omitted or set(active) != set(RISK_MODELS):
            raise ValueError("All four risk models must be present for shared support")
        # Match the full-risk evaluation cohort, including persistence in the
        # common-support check although only the other three models are reported.
        keep = common & sample.current_lots_available.gt(0)
        selected = sample.loc[keep].copy()
        selected["full"] = selected.actual_lots_available.eq(0).astype(int)
        for model in REPORT_MODELS:
            selected[model] = values.loc[keep, model]
            scores = risk_metrics(selected["full"], selected[model])
            target = reference.loc[
                reference.horizon_min.eq(horizon)
                & reference.event.eq("full")
                & reference.cohort.eq("open_at_departure")
                & reference.model.eq(model)
            ]
            if len(target) != 1:
                raise ValueError("Expected one published aggregate per model and horizon")
            target = target.iloc[0]
            if scores["n"] != target.n_scored or scores["events"] != target.events:
                raise ValueError("Per-park sample does not reproduce published counts")
            for metric in ("brier", "average_precision"):
                if not np.isclose(scores[metric], target[metric], rtol=1e-8, atol=1e-10):
                    raise ValueError(f"Pooled {model} {metric} does not reproduce evaluation")
        for park, rows in selected.groupby("carpark_id", observed=True, sort=True):
            row = {
                "horizon_min": horizon,
                "cohort": "open_at_departure",
                "carpark_id": park,
                "n_scored": len(rows),
                "full_events": int(rows.full.sum()),
                "full_event_rate": float(rows.full.mean()),
            }
            for model in REPORT_MODELS:
                metrics = risk_metrics(rows["full"], rows[model])
                row[f"{model}_brier"] = metrics["brier"]
                row[f"{model}_average_precision"] = metrics["average_precision"]
            result.append(row)
    return pd.DataFrame(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path,
                        default=DEFAULT_STUDY / "risk/risk_predictions.csv")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    table = summarize(args.predictions, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False, float_format="%.10g")
    print(table.groupby("horizon_min")[["n_scored", "full_events"]].sum().to_string())
    print(f"Saved {len(table)} carpark-horizon summaries to {args.output}")


if __name__ == "__main__":
    main()
