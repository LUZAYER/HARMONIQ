"""
Claim-Level GBDT Meta-Classifier (HistGradientBoostingClassifier).

Trains a gradient-boosted decision tree on claim-level signals:
Features: [sig_C, sig_R, sig_S, sig_V, avail_C, avail_R, avail_S, avail_V, claim_len, entity_count]
Leverages native NaN handling so splits handle missing/unretrieved evidence natively
without synthetic 0.5 or mean imputation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold


FEATURE_COLS = [
    "sig_C", "sig_R", "sig_S", "sig_V",
    "avail_C", "avail_R", "avail_S", "avail_V",
    "claim_len", "entity_count"
]


class ClaimGBDT:
    """HistGradientBoostingClassifier meta-classifier with native NaN support."""

    def __init__(
        self,
        max_iter: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        l2_regularization: float = 0.1,
        random_state: int = 42
    ):
        self.params = {
            "max_iter": max_iter,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "l2_regularization": l2_regularization,
            "random_state": random_state,
            "early_stopping": True,
            "n_iter_no_change": 10
        }
        self.model = HistGradientBoostingClassifier(**self.params)
        self.is_fitted = False

    def fit(self, claims_df: pd.DataFrame) -> "ClaimGBDT":
        """
        Fits GBDT on train split claims.
        """
        X = claims_df[FEATURE_COLS].values
        y = claims_df["claim_label"].values

        # If only one class is present in a tiny slice, fallback safely
        if len(np.unique(y)) < 2:
            self.model = None
            self.is_fitted = True
            return self

        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, claims_df: pd.DataFrame) -> np.ndarray:
        """
        Predicts P(claim is hallucinated) for each row.
        """
        if not self.is_fitted:
            raise RuntimeError("ClaimGBDT must be fitted before calling predict_proba.")

        if self.model is None:
            # Fallback if fit saw single class
            return np.full(len(claims_df), 0.5)

        X = claims_df[FEATURE_COLS].values
        # Return probability of class 1 (hallucinated)
        return self.model.predict_proba(X)[:, 1]

    def cross_validate(
        self,
        claims_df: pd.DataFrame,
        n_splits: int = 5,
        param_grid: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Group k-fold cross validation on training groups.
        """
        groups = claims_df["group"].values
        X = claims_df[FEATURE_COLS].values
        y = claims_df["claim_label"].values

        gkf = GroupKFold(n_splits=n_splits)
        scores = []

        from sklearn.metrics import roc_auc_score

        for train_idx, val_idx in gkf.split(X, y, groups):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
                continue

            clf = HistGradientBoostingClassifier(**self.params)
            clf.fit(X_tr, y_tr)
            preds = clf.predict_proba(X_val)[:, 1]
            try:
                scores.append(roc_auc_score(y_val, preds))
            except Exception:
                pass

        return {
            "cv_auc_mean": float(np.mean(scores)) if scores else 0.5,
            "cv_auc_std": float(np.std(scores)) if scores else 0.0
        }
