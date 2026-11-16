"""
Unit tests for Claim GBDT, Calibrators, and Aggregators.

Verifies:
- HistGradientBoostingClassifier handles NaNs natively without throwing errors or requiring constant imputation.
- Length-damped soft-max (HCS) dampens risk monotonically as claim count k grows compared to raw max.
- Isotonic regression preserves probability bounds in [0, 1].
- Operating threshold tuning chooses optimal Youden's J.
"""

import unittest
import numpy as np
import pandas as pd

from src.fusion.gbdt import ClaimGBDT, FEATURE_COLS
from src.fusion.calibrate import Calibrator, compute_ece
from src.fusion.aggregate import (
    aggregate_hcs, aggregate_max, aggregate_mean,
    aggregate_noisy_or, aggregate_power_mean,
    tune_operating_thresholds_on_dev
)


class TestFusionAndAggregation(unittest.TestCase):
    """Test suite for fusion classifiers and aggregators."""

    def test_gbdt_native_nan_support(self):
        """Verify that GBDT fits and predicts natively on datasets containing NaNs."""
        n_samples = 60
        # Create synthetic data with intentional NaNs in retrieval and selfverif signals
        data = {
            "sig_C": np.random.uniform(0, 1, n_samples),
            "sig_R": [np.nan if i % 2 == 0 else np.random.uniform(0, 1) for i in range(n_samples)],
            "sig_S": np.random.uniform(0, 1, n_samples),
            "sig_V": [np.nan if i % 3 == 0 else np.random.uniform(0, 1) for i in range(n_samples)],
            "avail_C": [1] * n_samples,
            "avail_R": [0 if i % 2 == 0 else 1 for i in range(n_samples)],
            "avail_S": [1] * n_samples,
            "avail_V": [0 if i % 3 == 0 else 1 for i in range(n_samples)],
            "claim_len": np.random.randint(5, 25, n_samples),
            "entity_count": np.random.randint(0, 5, n_samples),
            "claim_label": [0 if i < 30 else 1 for i in range(n_samples)],
            "group": [f"grp_{i//2}" for i in range(n_samples)]
        }
        df = pd.DataFrame(data)

        gbdt = ClaimGBDT(max_iter=20, max_depth=3)
        gbdt.fit(df)
        preds = gbdt.predict_proba(df)

        self.assertEqual(len(preds), n_samples)
        self.assertTrue(np.all(preds >= 0.0) and np.all(preds <= 1.0))
        self.assertFalse(np.any(np.isnan(preds)), "GBDT predictions must not contain NaN")

    def test_hcs_length_damping_mitigates_compounding_fpr(self):
        """Verify that HCS dampens score relative to raw max when k increases."""
        # Consider a response with 1 borderline claim and k-1 clean claims
        single_claim = np.array([0.70])
        multi_claims_5 = np.array([0.70, 0.10, 0.10, 0.10, 0.10])
        multi_claims_15 = np.array([0.70] + [0.10] * 14)

        score_1 = aggregate_hcs(single_claim, alpha=0.70, beta=0.10)
        score_5 = aggregate_hcs(multi_claims_5, alpha=0.70, beta=0.10)
        score_15 = aggregate_hcs(multi_claims_15, alpha=0.70, beta=0.10)

        # Raw max is invariant at 0.70
        self.assertEqual(aggregate_max(single_claim), aggregate_max(multi_claims_15))

        # HCS should be monotonically lower than raw max as k increases with low mean
        self.assertLess(score_5, score_1)
        self.assertLess(score_15, score_5)

    def test_aggregators_on_clean_and_hallucinated_responses(self):
        """Verify aggregators correctly handle all-factual vs all-hallucinated responses."""
        all_clean = np.array([0.05, 0.02, 0.08])
        all_hallu = np.array([0.95, 0.90, 0.88])

        self.assertLess(aggregate_max(all_clean), 0.15)
        self.assertGreater(aggregate_max(all_hallu), 0.85)

        self.assertLess(aggregate_mean(all_clean), 0.10)
        self.assertGreater(aggregate_mean(all_hallu), 0.85)

        self.assertLess(aggregate_noisy_or(all_clean), 0.20)
        self.assertGreater(aggregate_noisy_or(all_hallu), 0.95)

    def test_calibration_and_ece(self):
        """Verify isotonic calibrator and ECE calculation."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_uncal = np.array([0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        cal = Calibrator(method="isotonic")
        cal.fit(y_uncal, y_true)
        calibrated = cal.calibrate(y_uncal)

        self.assertTrue(np.all(calibrated >= 0.0) and np.all(calibrated <= 1.0))
        ece = compute_ece(y_true, calibrated, n_bins=4)
        self.assertLessEqual(ece, 0.25)

    def test_operating_threshold_tuning(self):
        """Verify Youden's J optimization on dev."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_scores = np.array([0.05, 0.12, 0.18, 0.35, 0.42, 0.48, 0.65, 0.78, 0.85, 0.92])

        res = tune_operating_thresholds_on_dev(y_true, y_scores)
        tau_star = res["tau_star"]
        self.assertTrue(0.35 <= tau_star <= 0.65, f"Tau* ({tau_star}) should cleanly separate classes")


if __name__ == '__main__':
    unittest.main()
