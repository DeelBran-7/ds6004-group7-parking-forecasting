import unittest

import pandas as pd

from scripts.analyze_rq_subgroups import subgroup_masks


class RqSubgroupTests(unittest.TestCase):
    def test_evening_interval_includes_start_but_excludes_end(self):
        times = pd.Series(pd.to_datetime(["2026-08-07 16:59", "2026-08-07 17:00", "2026-08-07 19:59", "2026-08-07 20:00"]))
        masks = subgroup_masks(times)
        self.assertEqual(masks["evening_peak"].tolist(), [False, True, True, False])
        self.assertTrue(masks["overall"].all())

    def test_weekend_uses_departure_day_and_can_overlap_evening(self):
        times = pd.Series(pd.to_datetime(["2026-08-07 23:45", "2026-08-08 00:00", "2026-08-09 18:00", "2026-08-10 00:00"]), index=[4, 5, 6, 7])
        masks = subgroup_masks(times)
        self.assertEqual(masks["weekend"].tolist(), [False, True, True, False])
        self.assertEqual((masks["weekend"] & masks["evening_peak"]).tolist(), [False, False, True, False])
        self.assertEqual(masks["weekend"].index.tolist(), [4, 5, 6, 7])

    def test_missing_or_timezone_ambiguous_inputs_are_rejected(self):
        for times in (pd.Series([pd.NaT]), pd.Series(pd.date_range("2026-08-08", periods=2, tz="UTC"))):
            with self.subTest(times=times), self.assertRaises(ValueError):
                subgroup_masks(times)
