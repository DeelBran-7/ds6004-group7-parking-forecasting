import unittest

import numpy as np
import pandas as pd

from evaluate_models import RISK_MODELS, evaluate_forecasts, evaluate_risk


def test_rows():
    return pd.DataFrame({
        "decision_time": pd.date_range("2026-08-24", periods=6),
        "horizon_min": 15, "carpark_id": "KAM", "split": ["test"] * 5 + ["validation"],
        "current_is_valid": True, "actual_is_observed": True,
        "forecast_label_eligible": [True, True, True, True, False, True],
        "actual_lots_available": [0, 10, 9, 2, 4, 100], "actual_total_lots": 100,
        "is_holiday": [0, 0, 0, 1, 0, 0],
    })


class OccupancyEvaluationTests(unittest.TestCase):
    def test_shared_rows_holiday_slice_and_exact_near_full_boundary(self):
        data = test_rows().assign(
            B0=[0.8, 0.95, 0.9, 0.9, 0.0, 1.0],
            B1=[0.8, 0.95, 0.9, np.nan, 0.0, 1.0],
            M0=[0.9, 0.9, 0.92, 0.9, 0.0, 1.0],
            M1=[0.9, 0.95, 0.92, 0.9, 0.0, 1.0],
        )
        data["actual_occupancy"] = 1 - data["actual_lots_available"] / data["actual_total_lots"]
        scores, comparisons = evaluate_forecasts(data, repetitions=100, seed=42)
        overall = scores.loc[scores["cohort"].eq("all")].set_index("model")
        self.assertTrue(overall["n_eligible"].eq(4).all())
        self.assertTrue(overall["n_scored"].eq(3).all())
        self.assertEqual(overall.at["M1", "n_available"], 4)
        self.assertAlmostEqual(overall.at["M1", "mae"], (0.1 + 0.05 + 0.01) / 3)
        self.assertEqual(overall.at["M1", "near_full_tp"], 1)
        self.assertEqual(overall.at["M1", "near_full_fp"], 1)  # Exactly 10% is not near-full.
        self.assertEqual(overall.at["M1", "near_full_fn"], 1)  # Forecast exactly .9 is negative.
        self.assertEqual(overall.at["M1", "near_full_precision"], 0.5)
        self.assertEqual(overall.at["M1", "near_full_recall"], 0.5)
        self.assertTrue(scores.loc[scores["cohort"].eq("near_full"), "near_full_precision"].isna().all())
        self.assertTrue(scores.loc[scores["cohort"].eq("holiday"), "status"].eq("no_shared_predictions").all())
        pair = comparisons.loc[comparisons["comparison"].eq("M1-B0") & comparisons["cohort"].eq("all")].iloc[0]
        self.assertEqual(pair["n_pairs"], 3)
        self.assertAlmostEqual(pair["estimate"], overall.at["M1", "mae"] - overall.at["B0", "mae"])

    def test_entirely_unavailable_weekly_baseline_is_explicitly_omitted(self):
        data = test_rows().assign(B0=0.7, B1=np.nan, M0=0.8, M1=0.9)
        data["actual_occupancy"] = 1 - data["actual_lots_available"] / data["actual_total_lots"]
        scores, _ = evaluate_forecasts(data, repetitions=100, seed=42)
        overall = scores.loc[scores["cohort"].eq("all")].set_index("model")
        self.assertTrue(overall["omitted_models"].eq("B1").all())
        self.assertEqual(overall.at["B1", "status"], "model_unavailable")
        self.assertEqual(overall.at["B1", "n_scored"], 0)
        self.assertEqual(overall.at["M0", "n_scored"], 4)


class RiskEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.data = test_rows().assign(current_lots_available=[5, 10, 0, 3, 4, 100], low_spaces_threshold=3)
        for model in RISK_MODELS:
            self.data[f"p_full_{model}"] = [0.8, 0.1, 0.2, 0.4, 0.0, 1.0]
            self.data[f"p_low_spaces_{model}"] = [0.8, 0.2, 0.3, 0.6, 0.0, 1.0]
        self.data.loc[1, "p_full_catboost"] = np.nan
        self.data["p_low_spaces_occupancy_proxy"] = np.nan

    def test_eligibility_event_matching_and_reliability_counts(self):
        scores, reliability, comparisons = evaluate_risk(self.data, repetitions=100, seed=42)
        main = scores.loc[scores["cohort"].eq("open_at_departure")].set_index(["event", "model"])
        self.assertEqual(main.at[("full", "xgboost"), "n_eligible"], 3)
        self.assertEqual(main.at[("full", "xgboost"), "n_scored"], 2)
        self.assertAlmostEqual(main.at[("full", "xgboost"), "brier"], (0.04 + 0.16) / 2)
        self.assertEqual(main.at[("low_spaces", "xgboost"), "events"], 2)
        self.assertEqual(main.at[("low_spaces", "xgboost"), "n_scored"], 3)
        self.assertEqual(main.at[("low_spaces", "occupancy_proxy"), "status"], "model_unavailable")
        bins = reliability.loc[reliability["event"].eq("full") & reliability["model"].eq("xgboost")
                               & reliability["cohort"].eq("open_at_departure")]
        self.assertEqual(len(bins), 10)
        self.assertEqual(int(bins["n"].sum()), 2)
        self.assertEqual(int(bins["events"].sum()), 1)
        absent = comparisons.loc[comparisons["event"].eq("low_spaces")
                                 & comparisons["comparison"].eq("xgboost-occupancy_proxy")]
        self.assertTrue(absent["n_pairs"].eq(0).all())

    def test_rejects_labels_that_disagree_with_observed_counts(self):
        with self.assertRaisesRegex(ValueError, "actual_full"):
            evaluate_risk(self.data.assign(actual_full=0), repetitions=100, seed=42)


if __name__ == "__main__":
    unittest.main()
