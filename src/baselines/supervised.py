"""
Supervised Single-Signal and Linear Baseline Classifiers.

Used to test whether HARMONIQ's performance gains stem from multi-signal fusion
or merely from in-domain supervised fine-tuning.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer


class SupervisedBaselines:
    """Trains single-signal GBDT baselines and multi-signal Logistic Regression."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.gbdt_semantic = HistGradientBoostingClassifier(random_state=random_state)
        self.gbdt_retrieval = HistGradientBoostingClassifier(random_state=random_state)
        self.imputer = SimpleImputer(strategy="constant", fill_value=0.5)
        self.logreg = LogisticRegression(random_state=random_state, max_iter=200)

    def fit(self, train_claims_df: pd.DataFrame):
        y = train_claims_df["claim_label"].values

        # 1. GBDT on MiniCheck/Semantic signal alone
        X_s = train_claims_df[["sig_S"]].values
        self.gbdt_semantic.fit(X_s, y)

        # 2. GBDT on Retrieval/NLI signal alone
        X_r = train_claims_df[["sig_R"]].values
        self.gbdt_retrieval.fit(X_r, y)

        # 3. Logistic Regression on all signals (with explicit NaN imputation)
        feature_cols = [
            "sig_C", "sig_R", "sig_S", "sig_V",
            "avail_C", "avail_R", "avail_S", "avail_V",
            "claim_len", "entity_count"
        ]
        X_all = train_claims_df[feature_cols].values
        X_all_imp = self.imputer.fit_transform(X_all)
        self.logreg.fit(X_all_imp, y)

    def predict_claims(self, claims_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        feature_cols = [
            "sig_C", "sig_R", "sig_S", "sig_V",
            "avail_C", "avail_R", "avail_S", "avail_V",
            "claim_len", "entity_count"
        ]
        X_all = claims_df[feature_cols].values
        X_all_imp = self.imputer.transform(X_all)

        return {
            "Supervised_GBDT_S": self.gbdt_semantic.predict_proba(claims_df[["sig_S"]].values)[:, 1],
            "Supervised_GBDT_R": self.gbdt_retrieval.predict_proba(claims_df[["sig_R"]].values)[:, 1],
            "Supervised_LR": self.logreg.predict_proba(X_all_imp)[:, 1]
        }
