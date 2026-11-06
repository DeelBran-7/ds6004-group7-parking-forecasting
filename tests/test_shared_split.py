import unittest

import pandas as pd

from forecast import split_times


class SharedSplitTests(unittest.TestCase):
    def test_adding_older_history_preserves_shared_splits_and_embargo(self):
        times = pd.Series(pd.date_range("2024-01-01", "2026-01-01", freq="15min"))
        recent = times.loc[times.ge("2025-01-01")]
        boundaries = ("2025-06-01", "2025-09-01")
        long_split = pd.Series(split_times(times, boundaries), index=times.index)
        short_split = pd.Series(split_times(recent, boundaries), index=recent.index)
        pd.testing.assert_series_equal(long_split.loc[recent.index], short_split)
        self.assertEqual(long_split.loc[times.eq("2025-05-31 23:30")].iat[0], "embargo")
        self.assertEqual(long_split.loc[times.eq("2025-08-31 23:45")].iat[0], "embargo")
        self.assertEqual(long_split.loc[times.eq("2025-09-01")].iat[0], "test")

    def test_invalid_shared_boundaries_fail(self):
        times = pd.Series(pd.date_range("2025-01-01", periods=100, freq="15min"))
        for boundaries in (("2025-01-01", "2025-01-02"), ("2025-01-01 09:01", "2025-01-02"),
                           ("2025-01-02", "2025-01-01"), ("2025-01-02", "2027-01-01")):
            with self.subTest(boundaries=boundaries), self.assertRaises(ValueError):
                split_times(times, boundaries)
