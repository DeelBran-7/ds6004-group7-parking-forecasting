import csv
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forecast import load_history, load_holidays, make_examples


COLUMNS = ("timestamp", "carpark_id", "total_lots", "lots_available", "lag_seconds")
START = datetime(2026, 11, 2, 8)


def at(minutes):
    return START + timedelta(minutes=minutes)


def observation(when, carpark="P1", free=75, capacity=100, lag=30):
    return when.isoformat(), carpark, capacity, free, lag


class ForecastDataTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.directory = Path(tmp.name)
        path = self.write_csv(
            "holidays.csv",
            ("date", "day", "holiday"),
            [
                ("2026-01-01", "Thursday", "New Year's Day"),
                ("2026-11-09", "Monday", "Deepavali (Observed)"),
                ("2026-12-25", "Friday", "Christmas Day"),
            ],
        )
        self.holidays = load_holidays(path)

    def write_csv(self, name, columns, rows):
        path = self.directory / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
        return path

    def examples(self, rows, horizon):
        path = self.write_csv("history.csv", COLUMNS, rows)
        history, _ = load_history(path)
        return make_examples(history, self.holidays, horizon)

    def row(self, frame, when, carpark="P1"):
        match = frame.loc[
            frame["decision_time"].eq(pd.Timestamp(when))
            & frame["carpark_id"].eq(carpark)
        ]
        self.assertEqual(len(match), 1, f"{carpark} at {when}")
        return match.iloc[0]

    def test_gaps_are_not_filled_by_next_row_or_other_carpark(self):
        rows = [
            observation(at(0), free=80),
            observation(at(30), free=70),
            observation(at(60), free=4),
            observation(at(75), free=62),
            *[observation(at(i * 15), "P2", free=20) for i in range(7)],
        ]
        at_30 = self.row(self.examples(rows, 15), at(30))
        self.assertEqual(at_30["target_time"], pd.Timestamp(at(45)))
        self.assertTrue(pd.isna(at_30["occ_lag_15"]))
        self.assertTrue(pd.isna(at_30["actual_occupancy"]))
        self.assertTrue(pd.isna(at_30["actual_lots_available"]))

        exact = self.row(self.examples(rows, 30), at(30))
        self.assertEqual(exact["target_time"], pd.Timestamp(at(60)))
        self.assertAlmostEqual(exact["actual_occupancy"], 0.96)
        self.assertEqual(exact["actual_lots_available"], 4)
        self.assertEqual(exact["actual_total_lots"], 100)

    def test_origin_features_are_unchanged_by_future_and_other_carpark(self):
        own = [observation(at(i * 15), free=80 - i * 10) for i in range(8)]
        other = [observation(at(i * 15), "P2", free=10) for i in range(8)]
        revised_own = own[:3] + [
            observation(at(i * 15), free=1) for i in range(3, 8)
        ]
        revised_other = [
            observation(at(i * 15), "P2", free=99) for i in range(8)
        ]
        initial = self.row(self.examples(own + other, 15), at(30))
        revised = self.row(self.examples(revised_own + revised_other, 15), at(30))

        self.assertAlmostEqual(initial["B0"], 0.4)
        self.assertAlmostEqual(initial["occ_lag_15"], 0.3)
        self.assertNotEqual(initial["actual_occupancy"], revised["actual_occupancy"])
        features = ["current_occupancy", "B0", "occ_lag_15", "occ_roll_mean_60"]
        pd.testing.assert_series_equal(
            initial[features], revised[features], check_names=False
        )

    def test_previous_week_baseline_matches_arrival_clock_time(self):
        week_later = at(0) + timedelta(days=7)
        rows = [
            observation(at(0), free=90),
            observation(at(15), free=65),
            observation(at(30), free=60),
            observation(week_later, free=20),
            observation(week_later, "P2", free=60),
            observation(week_later + timedelta(minutes=15), free=10),
            observation(week_later + timedelta(minutes=30), free=5),
        ]
        for horizon, expected in [(15, 0.35), (30, 0.4)]:
            with self.subTest(horizon=horizon):
                frame = self.examples(rows, horizon)
                first = self.row(frame, week_later)
                self.assertAlmostEqual(first["B1"], expected)
                self.assertEqual(first["is_holiday"], 1)
                self.assertTrue(pd.isna(self.row(frame, week_later, "P2")["B1"]))

    def test_splits_embargo_prevents_future_labels_crossing_boundaries(self):
        frame = self.examples(
            [observation(at(i * 15)) for i in range(20)],
            30,
        )
        for minute, expected in {
            135: "train",
            150: "embargo",
            165: "embargo",
            180: "validation",
            195: "validation",
            210: "embargo",
            225: "embargo",
            240: "test",
        }.items():
            with self.subTest(decision_time=at(minute)):
                self.assertEqual(self.row(frame, at(minute))["split"], expected)
        self.assertTrue(
            frame.loc[frame["split"].eq("train"), "target_time"].lt(at(180)).all()
        )
        self.assertTrue(
            frame.loc[frame["split"].eq("validation"), "target_time"].lt(at(240)).all()
        )

    def test_stale_and_invalid_target_counts_are_unavailable(self):
        rows = [
            observation(at(0)),
            observation(at(15), lag=901),
            observation(at(30)),
            observation(at(45), free=101),
            observation(at(60)),
            observation(at(75), free=3, capacity=120),
        ]
        frame = self.examples(rows, 15)
        self.assertTrue(pd.isna(self.row(frame, at(0))["actual_occupancy"]))
        self.assertTrue(pd.isna(self.row(frame, at(30))["occ_lag_15"]))
        self.assertTrue(pd.isna(self.row(frame, at(30))["actual_occupancy"]))
        self.assertAlmostEqual(self.row(frame, at(60))["actual_occupancy"], 0.975)

    def test_holiday_flags_use_singapore_arrival_date_and_preserve_unknown_year(self):
        rows = [
            ("2025-12-31T15:15:00Z", "P1", 100, 80, 30),
            ("2025-12-31T15:30:00Z", "P1", 100, 80, 30),
            ("2025-12-31T15:45:00Z", "P1", 100, 70, 30),
            observation(datetime(2026, 1, 1, 0, 0), free=40),
            observation(datetime(2026, 1, 1, 0, 15)),
            observation(datetime(2026, 1, 1, 0, 30)),
        ]
        frame = self.examples(rows, 15)
        unknown = self.row(frame, datetime(2025, 12, 31, 23, 30))
        self.assertTrue(pd.isna(unknown["is_holiday"]))
        midnight = self.row(frame, datetime(2025, 12, 31, 23, 45))
        self.assertEqual(midnight["target_time"], pd.Timestamp("2026-01-01"))
        self.assertEqual(midnight["is_holiday"], 1)
        self.assertAlmostEqual(midnight["actual_occupancy"], 0.6)

    def test_malformed_history_raises_clear_validation_error(self):
        cases = [
            ("missing field", COLUMNS[:-1], [observation(at(0))[:-1]]),
            ("duplicate carpark timestamp", COLUMNS, [observation(at(0))] * 2),
        ]
        for name, columns, rows in cases:
            with self.subTest(case=name):
                path = self.write_csv("bad.csv", columns, rows)
                with self.assertRaises(ValueError):
                    load_history(path)


if __name__ == "__main__":
    unittest.main()
