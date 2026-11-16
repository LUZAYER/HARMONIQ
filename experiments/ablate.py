#!/usr/bin/env python3
"""
Comprehensive Ablation Study Suite for HARMONIQ.

Ablations:
1. Signal stack: Leave-one-signal-out (w/o C, w/o R, w/o S, w/o V) & add-one-in (C only, R only, S only, V only)
2. Claim aggregation: Max, Mean, Noisy-OR, Power-Mean (p=2), HCS (with damping), HCS (w/o damping, beta=0)
3. Imputation mechanics: Native NaN branching vs. Mean Imputation vs. 0.5 Imputation
4. Availability flags: Flags enabled vs. flags removed

Outputs:
- results/ablation_results.csv
- results/ablation_results.md
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fusion.aggregate import (
    aggregate_hcs, aggregate_max, aggregate_mean,
    aggregate_noisy_or, aggregate_power_mean
)


def evaluate_ablation_variant(
    feature_cols: list,
    train_df: pd.DataFrame,
    test_claims_df: pd.DataFrame,
    aggregator_fn,
    impute_val: float = None
) -> dict:
    X_tr = train_df[feature_cols].values
    y_tr = train_df["claim_label"].values

    X_te = test_claims_df[feature_cols].values
    y_te_claims = test_claims_df["claim_label"].values

    if impute_val is not None:
        X_tr = np.nan_to_num(X_tr, nan=impute_val)
        X_te = np.nan_to_num(X_te, nan=impute_val)

    clf = HistGradientBoostingClassifier(max_iter=100, max_depth=4, random_state=42)
    clf.fit(X_tr, y_tr)
    preds = clf.predict_proba(X_te)[:, 1]

    temp_df = test_claims_df.copy()
    temp_df["prob"] = preds

    # Aggregate to response level
    resp_scores = []
    resp_labels = []
    for resp_id, grp in temp_df.groupby("response_id"):
        c_probs = grp["prob"].values
        resp_scores.append(aggregator_fn(c_probs))
        resp_labels.append(grp["response_label"].iloc[0])

    resp_scores = np.array(resp_scores)
    resp_labels = np.array(resp_labels)

    auc = roc_auc_score(resp_labels, resp_scores)
    y_pred = (resp_scores >= 0.5).astype(int)
    bal_acc = balanced_accuracy_score(resp_labels, y_pred)
    cm = confusion_matrix(resp_labels, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)

    return {
        "AUC": round(float(auc), 4),
        "Bal_Acc": round(float(bal_acc), 4),
        "FPR": round(float(fpr), 4),
        "FNR": round(float(fnr), 4)
    }


def run_ablations(
    claims_parquet: str = "results/claims.parquet",
    output_csv: str = "results/ablation_results.csv",
    output_md: str = "results/ablation_results.md"
):
    claims_df = pd.read_parquet(claims_parquet)
    train_df = claims_df[claims_df["split"] == "train"].copy()
    test_df = claims_df[claims_df["split"] == "test"].copy()

    all_features = [
        "sig_C", "sig_R", "sig_S", "sig_V",
        "avail_C", "avail_R", "avail_S", "avail_V",
        "claim_len", "entity_count"
    ]

    hcs_default = lambda p: aggregate_hcs(p, alpha=0.70, beta=0.05)
    hcs_nodamp = lambda p: aggregate_hcs(p, alpha=0.70, beta=0.0)

    rows = []

    # 1. Full HARMONIQ
    res_full = evaluate_ablation_variant(all_features, train_df, test_df, hcs_default)
    rows.append({"Configuration": "Full HARMONIQ", **res_full})

    # 2. Leave-One-Signal-Out
    # w/o Confidence (C)
    feats_no_c = [c for c in all_features if "sig_C" not in c and "avail_C" not in c]
    rows.append({"Configuration": "w/o Confidence (C)", **evaluate_ablation_variant(feats_no_c, train_df, test_df, hcs_default)})

    # w/o Retrieval (R)
    feats_no_r = [c for c in all_features if "sig_R" not in c and "avail_R" not in c]
    rows.append({"Configuration": "w/o Retrieval NLI (R)", **evaluate_ablation_variant(feats_no_r, train_df, test_df, hcs_default)})

    # w/o Semantic (S)
    feats_no_s = [c for c in all_features if "sig_S" not in c and "avail_S" not in c]
    rows.append({"Configuration": "w/o Semantic Faithfulness (S)", **evaluate_ablation_variant(feats_no_s, train_df, test_df, hcs_default)})

    # w/o Self-Verification (V)
    feats_no_v = [c for c in all_features if "sig_V" not in c and "avail_V" not in c]
    rows.append({"Configuration": "w/o Self-Verification (V)", **evaluate_ablation_variant(feats_no_v, train_df, test_df, hcs_default)})

    # 3. Add-One-In (Single Signals)
    rows.append({"Configuration": "Signal C alone", **evaluate_ablation_variant(["sig_C", "avail_C"], train_df, test_df, hcs_default)})
    rows.append({"Configuration": "Signal R alone", **evaluate_ablation_variant(["sig_R", "avail_R"], train_df, test_df, hcs_default)})
    rows.append({"Configuration": "Signal S alone", **evaluate_ablation_variant(["sig_S", "avail_S"], train_df, test_df, hcs_default)})
    rows.append({"Configuration": "Signal V alone", **evaluate_ablation_variant(["sig_V", "avail_V"], train_df, test_df, hcs_default)})

    # 4. Aggregator Ablations
    rows.append({"Configuration": "Aggregator: Max Pooling", **evaluate_ablation_variant(all_features, train_df, test_df, aggregate_max)})
    rows.append({"Configuration": "Aggregator: Mean Pooling", **evaluate_ablation_variant(all_features, train_df, test_df, aggregate_mean)})
    rows.append({"Configuration": "Aggregator: Noisy-OR", **evaluate_ablation_variant(all_features, train_df, test_df, aggregate_noisy_or)})
    rows.append({"Configuration": "Aggregator: Power-Mean (p=2)", **evaluate_ablation_variant(all_features, train_df, test_df, lambda p: aggregate_power_mean(p, 2.0))})
    rows.append({"Configuration": "HCS w/o Length Damping (beta=0)", **evaluate_ablation_variant(all_features, train_df, test_df, hcs_nodamp)})

    # 5. Imputation Mechanics
    rows.append({"Configuration": "Mean Imputation (w/o Native NaN)", **evaluate_ablation_variant(all_features, train_df, test_df, hcs_default, impute_val=0.5)})
    rows.append({"Configuration": "Zero Imputation (w/o Native NaN)", **evaluate_ablation_variant(all_features, train_df, test_df, hcs_default, impute_val=0.0)})

    res_df = pd.DataFrame(rows)
    # Compute delta relative to Full HARMONIQ
    base_auc = res_full["AUC"]
    base_bal = res_full["Bal_Acc"]
    res_df["Delta_AUC"] = (res_df["AUC"] - base_auc).round(4)
    res_df["Delta_BalAcc"] = (res_df["Bal_Acc"] - base_bal).round(4)

    res_df.to_csv(output_csv, index=False)
    print(f"✓ Saved ablation results to {output_csv}")

    # Generate Markdown summary
    md_content = "# HARMONIQ Component Ablation Study\n\n"
    md_content += res_df.to_markdown(index=False)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"✓ Saved markdown table to {output_md}")
    print("\n" + res_df[["Configuration", "AUC", "Delta_AUC", "Bal_Acc", "FPR"]].to_string(index=False))


if __name__ == '__main__':
    run_ablations()
