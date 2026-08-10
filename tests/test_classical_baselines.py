import unittest

import numpy as np

from scripts.train_classical_baselines import (
    calculate_metrics,
    longest_true_run,
    sequence_features,
    window_means,
)


class ClassicalBaselineTests(unittest.TestCase):
    def test_sequence_features_has_expected_35_finite_values(self) -> None:
        features = sequence_features("ACDEFGHIKLMNPQRSTVWY")
        self.assertEqual(len(features), 35)
        self.assertTrue(np.isfinite(list(features.values())).all())
        self.assertAlmostEqual(sum(features[f"fraction_{aa}"] for aa in "ACDEFGHIKLMNPQRSTVWY"), 1.0)

    def test_sequence_features_rejects_invalid_residues(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid amino-acid symbols"):
            sequence_features("ACD*")

    def test_longest_hydrophobic_run(self) -> None:
        self.assertEqual(longest_true_run(np.array([False, True, True, False, True])), 2)

    def test_short_sequence_window_uses_global_mean(self) -> None:
        values = np.array([1.0, 3.0, 5.0])
        np.testing.assert_allclose(window_means(values, 9), [3.0])

    def test_metrics_include_confusion_counts(self) -> None:
        metrics = calculate_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.7, 0.8, 0.4]))
        self.assertEqual(
            {key: metrics[key] for key in ("tn", "fp", "fn", "tp")},
            {"tn": 1, "fp": 1, "fn": 1, "tp": 1},
        )


if __name__ == "__main__":
    unittest.main()
