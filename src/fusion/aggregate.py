"""
Response-Level Claim Aggregators and Decision Thresholding.

Aggregators:
- max: max_j p(c_j)
- mean: mean_j p(c_j)
- noisy_or: 1 - prod_j (1 - p(c_j))
- power_mean: (mean_j p(c_j)^p)^(1/p) for p in {1, 2, 3, 5}
- HCS (Length-Damped Soft-Max):
    HCS = (alpha * max + (1 - alpha) * mean) / (1 + beta * ln(1 + k))
    where alpha in [0, 1] and beta >= 0 are selected strictly on DEV.

Thresholding Protocol:
- Maximize Youden's J = TPR - FPR on calibrated DEV scores (tau*).
- Operating point at 5% FPR.
- Operating point at 90% Recall.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.metrics import roc_curve


def aggregate_max(claim_probs: np.ndarray) -> float:
    if len(claim_probs) == 0:
        return 0.0
    return float(np.max(claim_probs))


def aggregate_mean(claim_probs: np.ndarray) -> float:
    if len(claim_probs) == 0:
        return 0.0
    return float(np.mean(claim_probs))


def aggregate_noisy_or(claim_probs: np.ndarray) -> float:
    if len(claim_probs) == 0:
        return 0.0
    p = np.clip(claim_probs, 0.0, 1.0 - 1e-6)
    return float(1.0 - np.prod(1.0 - p))


def aggregate_power_mean(claim_probs: np.ndarray, p: float = 2.0) -> float:
    if len(claim_probs) == 0:
        return 0.0
    if p == 1.0:
        return float(np.mean(claim_probs))
    probs = np.clip(claim_probs, 0.0, 1.0)
    return float(np.power(np.mean(np.power(probs, p)), 1.0 / p))


def aggregate_hcs(claim_probs: np.ndarray, alpha: float = 0.70, beta: float = 0.05) -> float:
    """
    HARMONIQ Length-Damped Soft-Max (HCS).
    Mitigates compounding False Positive Rate as response claim length k grows.
    """
    if len(claim_probs) == 0:
        return 0.0
    k = len(claim_probs)
    max_p = np.max(claim_probs)
    mean_p = np.mean(claim_probs)
    numerator = alpha * max_p + (1.0 - alpha) * mean_p
    denominator = 1.0 + beta * np.log(1.0 + k)
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def tune_hcs_hyperparameters_on_dev(
    dev_claims_df: pd.DataFrame,
    dev_labels: Dict[str, int]
) -> Tuple[float, float]:
    """
    Grid search for alpha in [0.4, 0.9] and beta in [0.0, 0.2] to maximize ROC-AUC on DEV set.
    """
    from sklearn.metrics import roc_auc_score

    resp_ids = list(dev_labels.keys())
    y_true = np.array([dev_labels[rid] for rid in resp_ids])

    if len(np.unique(y_true)) < 2:
        return 0.70, 0.05

    claims_by_resp = dev_claims_df.groupby("response_id")["prob"].apply(np.array).to_dict()

    best_auc = -1.0
    best_alpha, best_beta = 0.70, 0.05

    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for beta in [0.0, 0.02, 0.05, 0.10, 0.15]:
            scores = np.array([
                aggregate_hcs(claims_by_resp.get(rid, np.array([0.0])), alpha=alpha, beta=beta)
                for rid in resp_ids
            ])
            try:
                score_auc = roc_auc_score(y_true, scores)
                if score_auc > best_auc:
                    best_auc = score_auc
                    best_alpha, best_beta = alpha, beta
            except Exception:
                pass

    return best_alpha, best_beta


def tune_operating_thresholds_on_dev(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Dict[str, float]:
    """
    Finds Youden's optimal threshold (tau*), 5% FPR threshold, and 90% Recall threshold on DEV.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # 1. Youden's J = TPR - FPR
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    tau_star = float(thresholds[best_idx])

    # 2. Operating threshold at 5% FPR (FPR <= 0.05)
    fpr_5_indices = np.where(fpr <= 0.05)[0]
    tau_fpr5 = float(thresholds[fpr_5_indices[-1]]) if len(fpr_5_indices) > 0 else 0.90

    # 3. Operating threshold at 90% Recall (TPR >= 0.90)
    rec_90_indices = np.where(tpr >= 0.90)[0]
    tau_rec90 = float(thresholds[rec_90_indices[0]]) if len(rec_90_indices) > 0 else 0.10

    return {
        "tau_star": tau_star,
        "tau_fpr5": tau_fpr5,
        "tau_rec90": tau_rec90
    }
