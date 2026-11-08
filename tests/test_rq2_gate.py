"""Decision-only gate and validation-only selection regression checks."""
import unittest

import pandas as pd

from scripts.run_rq2_gate_experiment import GRID, rank_gate, select_threshold


class GateTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({"carpark_id": ["near", "middle", "far"], "distance_m": [100., 200., 300.],
                             "total_lots": [10, 20, 50], "current_lots_available": [2, 3, 25],
                             "forecast_occupancy": [.7, .5, .2], "actual_total_lots": [1000, 1, 1],
                             "actual_lots_available": [0, 0, 0]})

    def test_threshold_is_inclusive_and_safe_parks_use_distance_not_excess_space(self):
        ranked = rank_gate(self.frame(), "CURRENT_GATE", 3)
        self.assertEqual(ranked.carpark_id.tolist(), ["middle", "far", "near"])

    def test_forecast_uses_current_capacity_and_ignores_arrival_labels(self):
        frame = self.frame()
        before = rank_gate(frame, "FORECAST_GATE", 3)
        self.assertEqual(before.carpark_id.tolist(), ["near", "middle", "far"])
        self.assertAlmostEqual(before.estimated_free.iloc[0], 3)
        frame["actual_total_lots"] = [1, 5000, 5000]
        frame["actual_lots_available"] = [1, 4999, 4999]
        after = rank_gate(frame, "FORECAST_GATE", 3)
        pd.testing.assert_series_equal(before.estimated_free, after.estimated_free)
        self.assertEqual(before.carpark_id.tolist(), after.carpark_id.tolist())

    def test_no_passing_candidate_falls_back_to_most_estimated_spaces(self):
        frame = self.frame(); frame["current_lots_available"] = [1, 2, 2]
        self.assertEqual(rank_gate(frame, "CURRENT_GATE", 3).carpark_id.tolist(), ["middle", "far", "near"])

    def selection_frame(self):
        records = []
        for horizon in (15, 30):
            records.append(dict(strategy="C0", horizon_min=horizon, threshold=float("nan"), matched_cases=100,
                                top1_full_count=2, top1_low_spaces_count=3, mean_distance_m=260))
            for k in GRID:
                records.append(dict(strategy="CURRENT_GATE", horizon_min=horizon, threshold=k, matched_cases=100,
                                    top1_full_count=3 if k < 5 else 1, top1_low_spaces_count=4 if k < 5 else 2,
                                    mean_distance_m=200+k))
        return pd.DataFrame(records).assign(split="validation", cohort="open_at_departure")

    def test_selection_uses_static_validation_constraints_at_both_horizons(self):
        rows=self.selection_frame()
        k, table=select_threshold(rows)
        self.assertEqual(k, 5)
        rows.loc[(rows.horizon_min==30)&(rows.threshold==5),"top1_full_count"]=3
        self.assertEqual(select_threshold(rows)[0], 10)
        rows.loc[rows.strategy.eq("CURRENT_GATE"),"top1_full_count"]=4
        k, table=select_threshold(rows)
        self.assertEqual(k,3)
        self.assertTrue(table.fallback_used.all())

    def test_selector_rejects_test_or_forecast_gate_rows(self):
        rows=self.selection_frame(); rows.loc[0,"split"]="test"
        with self.assertRaises(ValueError):select_threshold(rows)
        rows=self.selection_frame(); rows.loc[1,"strategy"]="FORECAST_GATE"
        with self.assertRaises(ValueError):select_threshold(rows)


if __name__ == "__main__": unittest.main()
