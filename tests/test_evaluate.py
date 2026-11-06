"""Part C must preserve B's paired outcomes and reject incompatible inputs."""

import unittest

import pandas as pd

from evaluate import evaluate_recommendations, ranked_choices, validate_handoff
from recommendation import evaluate


def fixture():
    parts = []
    for date in ("2026-09-01", "2026-09-02"):
        origin = pd.Timestamp(f"{date} 12:00")
        parts.append(pd.DataFrame({
            "decision_time": origin, "target_time": origin + pd.Timedelta(minutes=15),
            "horizon_min": 15, "carpark_id": ["P1", "P2", "P3", "P4"],
            "distance_m": [100., 200., 300., 400.], "distance_normalized": [.1, .2, .3, .4],
            "current_is_valid": True, "current_occupancy": [.4, .5, .6, .7],
            "current_lots_available": [6, 5, 4, 3], "total_lots": 10,
            "forecast_occupancy": [.4, .5, .6, .7], "actual_is_observed": True,
            "actual_lots_available": [0, 0, 0, 5], "actual_total_lots": 10,
        }))
    candidates = pd.concat(parts, ignore_index=True)
    _, rankings = evaluate(candidates, ["A", "B", "C0", "C"], .5, 3, "visitor_eligible")
    test = candidates.assign(split="test", M1=candidates["forecast_occupancy"], forecast_label_eligible=True)
    validation = test.iloc[:1].copy()
    validation["decision_time"] = pd.Timestamp("2026-08-01 12:00")
    validation["target_time"] = pd.Timestamp("2026-08-01 12:15")
    validation["split"] = "validation"
    occupancy = pd.concat([validation, test], ignore_index=True)
    risk = test.assign(low_spaces_threshold=3)
    settings = {"low_spaces_threshold": 3, "selected_distance_weight": .5, "forecast_column": "M1"}
    return occupancy, risk, rankings, settings


class EvaluationHandoffTests(unittest.TestCase):
    def test_b_outcomes_and_only_returned_top_three_are_scored(self):
        occupancy, risk, rankings, settings = fixture()
        validate_handoff(occupancy, risk, rankings, settings)
        metrics, pairs = evaluate_recommendations(rankings, repetitions=100, seed=1)
        self.assertTrue(metrics["top1_full_rate"].eq(1).all())
        # P4 is free but is outside the returned top three.
        self.assertTrue(metrics["top3_any_available_rate"].eq(0).all())
        full = pairs.loc[pairs["metric"].eq("full_rate")].iloc[0]
        self.assertEqual(full["status"], "no_day_level_variation")
        self.assertTrue(pd.isna(full["p_value"]))

    def test_rejects_outcomes_or_forecasts_from_another_run(self):
        occupancy, risk, rankings, settings = fixture()
        for column, value in (("actual_lots_available", 1), ("forecast_occupancy", .99)):
            with self.subTest(column=column):
                altered = rankings.copy()
                altered.loc[altered.index[0], column] = value
                with self.assertRaisesRegex(ValueError, "Rankings disagree"):
                    validate_handoff(occupancy, risk, altered, settings)

    def test_rejects_unmatched_strategy_cases(self):
        _, _, rankings, _ = fixture()
        omit = rankings["strategy"].eq("C") & rankings["decision_time"].eq(pd.Timestamp("2026-09-02 12:00"))
        with self.assertRaisesRegex(ValueError, "identical cases"):
            ranked_choices(rankings.loc[~omit])

    def test_rejects_relabelled_full_arrivals(self):
        occupancy, risk, rankings, settings = fixture()
        rankings.loc[rankings.index[0], "full_at_arrival"] = False
        with self.assertRaisesRegex(ValueError, "inconsistent full_at_arrival"):
            validate_handoff(occupancy, risk, rankings, settings)


if __name__ == "__main__":
    unittest.main()
