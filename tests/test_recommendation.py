"""Decision-time ranking and visit eligibility for Part B."""

import unittest

import pandas as pd

from recommendation import evaluate, matched_cases, paired_differences, rank_candidates, visitor_open


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.candidates = pd.DataFrame(
            {
                "carpark_id": ["PARK_A", "PARK_B", "PARK_C"],
                "distance_m": [100.0, 500.0, 300.0],
                "distance_normalized": [0.1, 0.5, 0.3],
                "current_lots_available": [1, 60, 20],
                "current_occupancy": [0.95, 0.80, 0.30],
                "forecast_occupancy": [0.05, 0.60, 0.90],
                "p_full_xgboost": [0.90, 0.04, 0.20],
                "p_full_catboost": [0.40, 0.70, 0.03],
            }
        )

    def test_strategies_follow_their_distinct_decision_signals(self):
        expected = {
            "A": "PARK_A",
            "B": "PARK_B",
            "C0": "PARK_C",
            "C": "PARK_A",
            "C_XGB_RISK": "PARK_B",
            "C_CAT_RISK": "PARK_C",
        }
        for strategy, top_choice in expected.items():
            with self.subTest(strategy=strategy):
                ranked = rank_candidates(self.candidates, strategy, 0.2)
                self.assertEqual(ranked["carpark_id"].iloc[0], top_choice)
                self.assertCountEqual(ranked["carpark_id"], self.candidates["carpark_id"])

    def test_weight_endpoints_isolate_distance_and_signal(self):
        for strategy in ("C0", "C", "C_XGB_RISK", "C_CAT_RISK"):
            with self.subTest(strategy=strategy):
                ranked = rank_candidates(self.candidates, strategy, 1.0)
                self.assertEqual(ranked["carpark_id"].tolist(), ["PARK_A", "PARK_C", "PARK_B"])
        self.assertEqual(
            rank_candidates(self.candidates, "C0", 0.0)["carpark_id"].tolist(),
            ["PARK_C", "PARK_B", "PARK_A"],
        )
        self.assertEqual(
            rank_candidates(self.candidates, "C", 0.0)["carpark_id"].tolist(),
            ["PARK_A", "PARK_B", "PARK_C"],
        )

    def test_ties_break_on_distance_then_id_regardless_of_input_order(self):
        candidates = pd.DataFrame(
            {
                "carpark_id": ["FAR", "NEAR_Z", "NEAR_A"],
                "distance_m": [500.0, 250.0, 250.0],
                "distance_normalized": [0.5, 0.25, 0.25],
                "current_lots_available": [10, 10, 10],
                "current_occupancy": [0.25, 0.5, 0.5],
                "forecast_occupancy": [0.25, 0.5, 0.5],
            }
        )
        for strategy in ("A", "B", "C0", "C"):
            with self.subTest(strategy=strategy):
                ranked = rank_candidates(candidates, strategy, 0.5)
                self.assertEqual(ranked["carpark_id"].tolist(), ["NEAR_A", "NEAR_Z", "FAR"])

    def test_changing_arrival_outcomes_does_not_change_rankings(self):
        first = self.candidates.assign(
            actual_lots_available=[0, 7, 60],
            actual_occupancy=[1.0, 0.9, 0.4],
            actual_is_observed=[True, True, True],
        )
        second = self.candidates.assign(
            actual_lots_available=[90, 0, float("nan")],
            actual_occupancy=[0.1, 1.0, float("nan")],
            actual_is_observed=[True, True, False],
        )
        for strategy in ("A", "B", "C0", "C", "C_XGB_RISK", "C_CAT_RISK"):
            with self.subTest(strategy=strategy):
                before = rank_candidates(first, strategy, 0.5)["carpark_id"].tolist()
                after = rank_candidates(second, strategy, 0.5)["carpark_id"].tolist()
                self.assertEqual(before, after)


class VisitorEligibilityTests(unittest.TestCase):
    def test_arrival_windows_include_opening_and_exclude_closing(self):
        cases = [
            ("7AM-10.30PM", "06:59", False),
            ("7AM-10.30PM", "07:00", True),
            ("7AM-10.30PM", "22:29", True),
            ("7AM-10.30PM", "22:30", False),
            ("7AM-7PM", "18:59", True),
            ("7AM-7PM", "19:00", False),
            ("WHOLE DAY", "00:00", True),
            ("NO", "12:00", False),
        ]
        for hours, clock_time, expected in cases:
            with self.subTest(hours=hours, clock_time=clock_time):
                arrival = pd.Timestamp(f"2026-11-02 {clock_time}")
                self.assertEqual(visitor_open(hours, arrival), expected)

    def test_unknown_visitor_hours_are_not_assumed_open(self):
        with self.assertRaises(ValueError):
            visitor_open("UNVERIFIED", pd.Timestamp("2026-11-02 12:00"))


class MatchedReplayTests(unittest.TestCase):
    def test_missing_eligible_park_excludes_case_but_closed_park_does_not(self):
        metadata = pd.DataFrame({"carpark_id": ["A", "B"], "pilot_candidate": [True, True],
                                 "short_term_parking": ["WHOLE DAY", "7AM-7PM"]})
        origins = pd.to_datetime(["2026-11-02 18:15", "2026-11-02 18:30"])
        forecasts = pd.DataFrame({"decision_time": origins, "target_time": origins + pd.Timedelta(minutes=30),
                                  "horizon_min": 30, "carpark_id": "A", "split": "test",
                                  "current_is_valid": True, "current_lots_available": 4,
                                  "current_occupancy": 0.6, "total_lots": 10, "actual_is_observed": True,
                                  "actual_lots_available": 3, "actual_total_lots": 10,
                                  "forecast_occupancy": 0.5})
        included, audit = matched_cases(forecasts, metadata)
        self.assertEqual(audit["expected_candidates"].tolist(), [2, 1])
        self.assertEqual(audit["missing_origin"].tolist(), [1, 0])
        self.assertEqual(audit["complete"].tolist(), [False, True])
        self.assertEqual(included["decision_time"].tolist(), [origins[1]])

    def test_paired_replay_scores_actual_arrivals_and_extra_distance(self):
        candidates = pd.DataFrame({"decision_time": pd.Timestamp("2026-11-02 09:00"), "horizon_min": 15,
                                   "carpark_id": ["A", "B"], "distance_m": [100.0, 200.0],
                                   "distance_normalized": [0.1, 0.2], "current_occupancy": [0.2, 0.7],
                                   "forecast_occupancy": [0.9, 0.2], "current_lots_available": [8, 3],
                                   "actual_total_lots": [10, 10], "actual_lots_available": [0, 2]})
        summary, rankings = evaluate(candidates, ["C0", "C"], 0.5, 3, "visitor_eligible")
        c = summary.set_index("strategy").loc["C"]
        self.assertEqual(c["top1_full_count"], 0)
        self.assertEqual(c["top1_low_spaces_count"], 1)
        self.assertEqual(c["mean_extra_vs_nearest_m"], 100.0)
        comparison = paired_differences(rankings, ["C"]).iloc[0]
        self.assertEqual(comparison["full_rate_delta"], -1.0)
        self.assertEqual(comparison["full_avoided"], 1)


if __name__ == "__main__":
    unittest.main()
