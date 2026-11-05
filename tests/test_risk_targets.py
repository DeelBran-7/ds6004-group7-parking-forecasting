import unittest

import numpy as np
import pandas as pd

from forecast_risk import add_risk_targets, reconcile_nested


class RiskTargetTests(unittest.TestCase):
    def test_separate_absolute_thresholds_and_missing_arrivals(self):
        examples = pd.DataFrame({
            "actual_is_observed": [True, True, True, True, False],
            "actual_lots_available": [0.0, 2.0, 3.0, 60.0, np.nan],
            "actual_total_lots": [20, 20, 20, 600, np.nan],
            "current_occupancy": [0.5] * 5,
            "occ_lag_15": [0.5] * 5,
        })
        add_risk_targets(examples, 3)
        np.testing.assert_equal(examples["actual_full"].to_numpy(), [1, 0, 0, 0, np.nan])
        np.testing.assert_equal(examples["actual_low_spaces"].to_numpy(), [1, 1, 0, 0, np.nan])

    def test_reconcile_keeps_full_risk_and_does_not_invent_missing_values(self):
        full, low, changed = reconcile_nested([0.7, 0.2, np.nan, 0.5], [0.5, 0.6, 0.5, np.nan])
        np.testing.assert_equal(full, [0.7, 0.2, np.nan, 0.5])
        np.testing.assert_equal(low, [0.7, 0.6, 0.5, np.nan])
        self.assertEqual(changed, 1)


if __name__ == "__main__":
    unittest.main()
