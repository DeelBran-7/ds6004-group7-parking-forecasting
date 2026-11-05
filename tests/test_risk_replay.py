import unittest

import numpy as np
import pandas as pd

from risk_replay import summarize_replay


def record(origin, park, current, arrival, full_risk, low_risk, *, horizon=15, **updates):
    return {
        "decision_time": pd.Timestamp(origin), "carpark_id": park, "horizon_min": horizon,
        "current_is_valid": True, "current_lots_available": current, "current_occupancy": 1 - current / 20,
        "actual_is_observed": True, "actual_lots_available": arrival, "low_spaces_threshold": 3,
        "p_full_xgboost": full_risk, "p_low_spaces_xgboost": low_risk, **updates,
    }


class ReplayTests(unittest.TestCase):
    def test_replay_keeps_full_and_low_outcomes_distinct_and_uses_open_candidates(self):
        when = "2026-11-02 10:00"
        frame = pd.DataFrame([
            record(when, "A", 10, 0, 0.60, 0.70),
            record(when, "B", 8, 2, 0.25, 0.60),
            record(when, "C", 0, 5, 0.00, 0.01),
            record(when, "A", 10, 3, 0.10, 0.20, horizon=30),
            record(when, "B", 8, 0, 0.50, 0.70, horizon=30),
            record(when, "C", 0, 0, 0.30, 0.60, horizon=30),
        ])
        transitions, replay = summarize_replay(frame, ["A", "B", "C"])
        t = transitions.set_index("horizon_min").loc[15]
        self.assertEqual(t["vacancy_declined_count"], 2)
        self.assertEqual(t["open_at_departure_count"], 2)
        self.assertEqual(t["open_to_full_rate"], 0.5)
        self.assertEqual(t["safe_to_low_rate"], 1.0)
        outcomes = replay.set_index(["horizon_min", "strategy"])
        self.assertEqual(outcomes.loc[(15, "most_current_free"), "first_full_count"], 1)
        ranked = outcomes.loc[(15, "xgboost_full_risk")]
        self.assertEqual(ranked["matched_origins"], 1)
        self.assertEqual(ranked["first_full_count"], 0)
        self.assertEqual(ranked["first_low_spaces_count"], 1)  # B has two spaces, not zero.
        self.assertEqual(ranked["top_three_any_available_count"], 1)
        self.assertEqual(outcomes.loc[(30, "xgboost_full_risk"), "first_low_spaces_count"], 0)

    def test_missing_candidate_score_or_arrival_excludes_origin_for_every_strategy(self):
        a, b, c = "2026-11-02 10:00", "2026-11-02 10:15", "2026-11-02 10:30"
        frame = pd.DataFrame([
            record(a, "A", 7, 0, 0.4, 0.8), record(a, "B", 4, 6, np.nan, 0.4),
            record(b, "A", 4, 0, 0.5, 0.8),
            record(b, "B", 5, 0, 0.3, 0.6, actual_is_observed=False),
            record(c, "A", 5, 0, 0.1, 0.7), record(c, "B", 4, 7, 0.1, 0.4),
            record(c, "C", 4, 7, 0.1, 0.4),
        ])
        frame["p_full_xgboost"] = frame["p_full_xgboost"].astype("Float64")
        transitions, replay = summarize_replay(frame, ["A", "B"])
        self.assertEqual(transitions.iloc[0]["matched_rows"], 5)
        self.assertEqual(set(replay["matched_origins"]), {1})
        outcomes = replay.set_index("strategy")
        self.assertEqual(outcomes.loc["most_current_free", "first_full_count"], 1)
        self.assertEqual(outcomes.loc["xgboost_full_risk", "first_full_count"], 0)
        # A requested candidate missing entirely prevents declaring an eligible comparison.
        _, no_match = summarize_replay(frame, ["A", "B", "C", "D"])
        self.assertEqual(set(no_match["matched_origins"]), {0})
        self.assertTrue(no_match["first_full_rate"].isna().all())


if __name__ == "__main__":
    unittest.main()
