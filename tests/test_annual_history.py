"""Regression checks for the segmented annual data contract."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forecast import HolidayCalendar, load_history, make_examples


START = pd.Timestamp("2025-11-05 00:00:00")
FIELDS = [
    "timestamp_sgt", "carpark_id", "lot_type", "total_lots_reported",
    "available_lots_reported", "lag_seconds", "snapshot_lag_seconds",
    "valid_5min", "segment_id", "target_available_15min",
    "target_occupancy_15min", "can_train_15min",
]


def row(slot, *, segment="KAM:0001", free=75, source_age=30, snapshot_age=15, valid=True):
    return {
        "timestamp_sgt": (START + pd.Timedelta(minutes=15 * slot)).isoformat(),
        "carpark_id": "KAM",
        "lot_type": "C",
        "total_lots_reported": 100,
        "available_lots_reported": free,
        "lag_seconds": source_age,
        "snapshot_lag_seconds": snapshot_age,
        "valid_5min": valid,
        "segment_id": segment,
        # Contradict the valid next observation and its eligibility on purpose.
        "target_available_15min": 98,
        "target_occupancy_15min": 0.02,
        "can_train_15min": False,
    }


class AnnualHistoryTests(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.path = Path(temp_dir.name) / "annual.csv"
        self.holidays = HolidayCalendar(frozenset(), frozenset({2025}))

    def load(self, rows):
        with self.path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return load_history(self.path)

    def test_recomputes_targets_and_eligibility_from_observations(self):
        history, audit = self.load([row(slot, free=2 if slot == 1 else 75) for slot in range(6)])
        examples = make_examples(history, self.holidays, 15)

        self.assertEqual(audit["source_schema"], "annual_segmented")
        self.assertTrue(examples.loc[0, "forecast_label_eligible"])
        self.assertEqual(examples.loc[0, "actual_lots_available"], 2)
        self.assertAlmostEqual(examples.loc[0, "actual_occupancy"], 0.98)
        self.assertAlmostEqual(examples.loc[0, "current_occupancy"], 0.25)

    def test_both_ages_and_supplier_quality_control_observation_validity(self):
        history, _ = self.load([
            row(0, source_age=300, snapshot_age=299),
            row(1, source_age=301, snapshot_age=15),
            row(2, source_age=30, snapshot_age=301),
            row(3, source_age=30, snapshot_age=None),
            row(4, source_age=30, snapshot_age=15, valid=False),
            row(5, segment=None, source_age=30, snapshot_age=15),
            row(6, segment="KAM:0002", source_age=22, snapshot_age=42),
        ])

        self.assertEqual(history["is_valid"].tolist(), [True, False, False, False, False, False, True])
        self.assertEqual(history.loc[0, "lag_seconds"], 300)
        self.assertEqual(history.loc[6, "lag_seconds"], 42)
        self.assertTrue(history.loc[1:5, "occupancy"].isna().all())

    def test_segment_break_excludes_training_label_but_preserves_observed_arrival(self):
        history, _ = self.load([
            row(0, free=80),
            row(1, free=75),
            row(2, segment=None, valid=False),
            row(3, segment="KAM:0002", free=3),
            row(4, segment="KAM:0002", free=2),
            row(5, segment="KAM:0002", free=1),
        ])
        examples = make_examples(history, self.holidays, 30)

        # At slot 1, a real arrival snapshot exists at slot 3, but slot 2 is
        # invalid, so this pair can be replayed without entering model fitting.
        self.assertTrue(examples.loc[1, "actual_is_observed"])
        self.assertEqual(examples.loc[1, "actual_lots_available"], 3)
        self.assertFalse(examples.loc[1, "forecast_label_eligible"])
        self.assertFalse(examples.loc[0, "actual_is_observed"])
        self.assertTrue(examples.loc[3, "forecast_label_eligible"])
        self.assertTrue(pd.isna(examples.loc[3, "occ_lag_30"]))
        self.assertEqual(examples.loc[3, "occ_roll_count_60"], 1)
        self.assertAlmostEqual(examples.loc[4, "occ_lag_15"], 0.97)


if __name__ == "__main__":
    unittest.main()
