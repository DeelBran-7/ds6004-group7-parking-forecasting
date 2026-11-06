import unittest

from evaluation_stats import holm_adjust, paired_day_inference


class PairedDayInferenceTests(unittest.TestCase):
    def test_unequal_day_sizes_keep_observation_weighting(self):
        # The two day means (+1, -1) average to zero, but the paired
        # observation-level mean is +0.5 because the first day is larger.
        result = paired_day_inference(["d1", "d1", "d1", "d2"], [1, 1, 1, 0], [0, 0, 0, 1],
                                      repetitions=1024, seed=7)
        self.assertEqual(result["status"], "ok")
        self.assertEqual((result["n_pairs"], result["n_days"]), (4, 2))
        self.assertEqual(result["candidate_mean"], 0.75)
        self.assertEqual(result["reference_mean"], 0.25)
        self.assertEqual(result["estimate"], 0.5)
        self.assertEqual((result["ci_low"], result["ci_high"]), (-1.0, 1.0))
        self.assertEqual(result["p_value"], 1.0)

    def test_paired_sign_flip_is_seeded_and_detects_consistent_direction(self):
        days = list(range(12))
        first = paired_day_inference(days, [0] * 12, [1] * 12, repetitions=2048, seed=83)
        second = paired_day_inference(days, [0] * 12, [1] * 12, repetitions=2048, seed=83)
        self.assertEqual(first, second)
        self.assertEqual((first["estimate"], first["ci_low"], first["ci_high"]), (-1, -1, -1))
        self.assertGreater(first["p_value"], 0)
        self.assertLess(first["p_value"], 0.01)

    def test_identical_outcomes_cannot_establish_equivalence(self):
        result = paired_day_inference([1, 2, 3], [0, 0, 0], [0, 0, 0])
        self.assertEqual(result["status"], "no_day_level_variation")
        self.assertEqual(result["estimate"], 0.0)
        self.assertIsNone(result["p_value"])
        self.assertIsNone(result["ci_low"])
        self.assertIsNone(result["ci_high"])

    def test_missing_pairs_and_one_day_do_not_produce_inference(self):
        empty = paired_day_inference([], [], [])
        self.assertEqual(empty["status"], "no_pairs")
        self.assertIsNone(empty["estimate"])
        one_day = paired_day_inference([1, 1], [1, 1], [0, 0])
        self.assertEqual(one_day["status"], "insufficient_days")
        self.assertEqual(one_day["estimate"], 1)
        self.assertIsNone(one_day["p_value"])

    def test_rejects_invalid_pair_vectors(self):
        for days, candidate, reference in (([1, 2], [1], [0]),
                                           ([1], [float("nan")], [0]),
                                           ([None], [1], [0]),
                                           ([[1]], [1], [0])):
            with self.subTest(days=days, candidate=candidate), self.assertRaises(ValueError):
                paired_day_inference(days, candidate, reference)

    def test_holm_preserves_order_and_unavailable_values(self):
        adjusted = holm_adjust([0.04, None, 0.01, 0.03, 0.8])
        self.assertEqual(adjusted, [0.09, None, 0.04, 0.09, 0.8])
        self.assertEqual(holm_adjust([None]), [None])
        self.assertEqual(holm_adjust([0.9, 1]), [1.0, 1.0])
        with self.assertRaises(ValueError):
            holm_adjust([float("nan")])


if __name__ == "__main__":
    unittest.main()
