"""
Probability Calibration and Reliability Diagnostics.

Fits Isotonic Regression and Platt Scaling (Logistic) strictly on DEV set.
Computes:
- Fixed-bin Expected Calibration Error (ECE)
- Adaptive-bin (quantile) ECE
- Brier Score
"""

import numpy as np
from typing import Tuple, Dict, Any
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    adaptive: bool = False
) -> float:
    """
    Computes Expected Calibration Error (ECE).
    """
    if len(y_true) == 0:
        return 0.0

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if adaptive:
        # Quantile binning: each bin has roughly equal count
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.quantile(y_prob, quantiles)
        bins[0] -= 1e-6
        bins[-1] += 1e-6
    else:
        bins = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        idx = (y_prob > bins[i]) & (y_prob <= bins[i + 1])
        if np.sum(idx) > 0:
            bin_acc = np.mean(y_true[idx])
            bin_conf = np.mean(y_prob[idx])
            ece += (np.sum(idx) / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


class Calibrator:
    """Isotonic and Platt Calibrator trained strictly on dev set."""

    def __init__(self, method: str = "isotonic"):
        self.method = method
        if method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        elif method == "platt":
            self.model = LogisticRegression()
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        self.is_fitted = False

    def fit(self, y_score: np.ndarray, y_true: np.ndarray) -> "Calibrator":
        """Fit calibrator on DEV scores and labels."""
        y_score = np.asarray(y_score)
        y_true = np.asarray(y_true)

        if self.method == "isotonic":
            self.model.fit(y_score, y_true)
        else:
            self.model.fit(y_score.reshape(-1, 1), y_true)

        self.is_fitted = True
        return self

    def calibrate(self, y_score: np.ndarray) -> np.ndarray:
        """Calibrates unseen scores."""
        if not self.is_fitted:
            return np.asarray(y_score)

        y_score = np.asarray(y_score)
        if self.method == "isotonic":
            return np.clip(self.model.predict(y_score), 0.0, 1.0)
        else:
            return self.model.predict_proba(y_score.reshape(-1, 1))[:, 1]

    def evaluate_calibration(self, y_score: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """Returns calibration diagnostics."""
        return {
            "ece_fixed": compute_ece(y_true, y_score, n_bins=10, adaptive=False),
            "ece_adaptive": compute_ece(y_true, y_score, n_bins=10, adaptive=True),
            "brier_score": float(brier_score_loss(y_true, y_score))
        }
