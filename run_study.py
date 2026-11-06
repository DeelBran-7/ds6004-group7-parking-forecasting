"""Run the A forecast, B replay and C evaluation as one reproducible study."""

import argparse
from pathlib import Path

import evaluate
import forecast
import forecast_risk
import recommendation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--holidays", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-lag-seconds", type=float, default=900)
    parser.add_argument("--low-spaces-threshold", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--validation-start", help="Optional shared validation boundary for history comparisons")
    parser.add_argument("--test-start", help="Optional shared test boundary, paired with --validation-start")
    args = parser.parse_args()
    if bool(args.validation_start) != bool(args.test_start):
        parser.error("Provide both --validation-start and --test-start")
    boundaries = (args.validation_start, args.test_start) if args.validation_start else None
    if args.repetitions < 1 or args.seed < 0 or args.low_spaces_threshold < 1:
        parser.error("Repetitions and low-space threshold must be positive; seed must be nonnegative")
    base = args.output
    occupancy_dir, risk_dir, rec_dir, eval_dir = (base / stage for stage in
                                                  ("forecast", "risk", "recommendation", "evaluation"))
    try:
        print("A: fitting occupancy and arrival-risk models", flush=True)
        forecast.run(args.history, args.holidays, occupancy_dir, max_lag_seconds=args.max_lag_seconds,
                     split_boundaries=boundaries)
        forecast_risk.run(args.history, args.holidays, risk_dir, threshold=args.low_spaces_threshold,
                          max_lag_seconds=args.max_lag_seconds, split_boundaries=boundaries)
        predictions, risk = occupancy_dir / "predictions.csv", risk_dir / "risk_predictions.csv"
        print("B: replaying matched recommendations", flush=True)
        recommendation.run(predictions, args.metadata, args.destination, rec_dir, risk_path=risk,
                           low_spaces_threshold=args.low_spaces_threshold)
        print("C: scoring forecasts, risks and paired policy outcomes", flush=True)
        evaluate.run(predictions, risk, rec_dir, eval_dir, repetitions=args.repetitions, seed=args.seed,
                     figures=not args.no_figures)
    except ValueError as error:
        parser.error(str(error))
    print(f"A–B–C study complete: {base}")


if __name__ == "__main__":
    main()
