import unittest

import numpy as np

from risk_calibration import fit_calibrator, reliability_bins, risk_metrics


class TestRiskCalibration(unittest.TestCase):
    def test_platt_corrects_deliberate_overconfidence_without_reversing_rank(self):
        raw = np.repeat([0.05, 0.95], 100)
        labels = np.concatenate([np.r_[np.ones(20), np.zeros(80)], np.r_[np.ones(80), np.zeros(20)]])
        calibrator = fit_calibrator(raw, labels)
        self.assertEqual(calibrator.status, "platt")
        result = calibrator.predict([0, 0.05, 0.95, 1])
        self.assertTrue(np.all(np.diff(result) > 0))
        self.assertAlmostEqual(result[1], 0.20, places=2)
        self.assertAlmostEqual(result[2], 0.80, places=2)

    def test_identity_status_when_calibration_cannot_be_supported(self):
        cases = [
            ([], [], "identity_no_rows"),
            ([0.1, 0.9], [0, 1], "identity_insufficient_classes"),
            ([0.5] * 10, [0, 1] * 5, "identity_constant_scores"),
            ([0.9] * 5 + [0.1] * 5, [0] * 5 + [1] * 5, "identity_nonpositive_slope"),
        ]
        for raw, labels, status in cases:
            with self.subTest(status=status):
                calibrator = fit_calibrator(raw, labels)
                self.assertEqual(calibrator.status, status)
                np.testing.assert_array_equal(calibrator.predict([0, 0.2, 1]), [0, 0.2, 1])

    def test_decile_boundaries_include_ten_percent_and_one(self):
        bins = reliability_bins([0, 1, 0, 1, 0], [0, np.nextafter(0.1, 0), 0.1, 0.9, 1])
        self.assertEqual([row["n"] for row in bins], [2, 1, 0, 0, 0, 0, 0, 0, 0, 2])
        self.assertEqual(bins[0]["event_rate"], 0.5)
        self.assertEqual(bins[1]["mean_predicted"], 0.1)
        self.assertIsNone(bins[2]["event_rate"])
        self.assertEqual(bins[9]["event_rate"], 0.5)

    def test_metrics_distinguish_empty_and_one_class_slices(self):
        empty = risk_metrics([], [])
        self.assertEqual(empty["n"], 0)
        self.assertIsNone(empty["brier"])
        metrics = risk_metrics([0, 0], [0.1, 0.2])
        self.assertAlmostEqual(metrics["brier"], 0.025)
        self.assertIsNotNone(metrics["log_loss"])
        self.assertIsNone(metrics["average_precision"])
        self.assertIsNone(metrics["roc_auc"])
        self.assertTrue(np.isfinite(risk_metrics([1, 0], [0, 1])["log_loss"]))

    def test_rejects_invalid_labels_probabilities_and_lengths(self):
        for labels, scores in [([1], [np.nan]), ([0], [1.1]), ([1], [-0.1]), ([2], [0.5]), ([1, 0], [0.5])]:
            with self.subTest(labels=labels, scores=scores):
                with self.assertRaises(ValueError):
                    risk_metrics(labels, scores)


if __name__ == "__main__":
    unittest.main()
