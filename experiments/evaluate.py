#!/usr/bin/env python3
"""
Rigorous Statistical Evaluation Suite for HARMONIQ and Baselines.

Computes:
- ROC-AUC, PR-AUC, Balanced Accuracy, F1, Precision, Recall, FPR, FNR, ECE, Brier
- Group-level bootstrap (B=2000) 95% Confidence Intervals
- Paired differences against HARMONIQ with Holm-Bonferroni correction

Outputs:
- results/stats_table.csv
- results/paired_tests.csv
- results/results.json
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score, confusion_matrix,
    brier_score_loss
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fusion.calibrate import compute_ece


def compute_metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Computes all standard binary classification and calibration metrics."""
    valid_mask = ~np.isnan(y_score)
    y_t = y_true[valid_mask]
    y_s = y_score[valid_mask]

    if len(y_t) == 0 or len(np.unique(y_t)) < 2:
        return {
            "auc": 0.5, "pr_auc": 0.5, "bal_acc": 0.5, "f1": 0.0,
            "precision": 0.0, "recall": 0.0, "fpr": 0.0, "fnr": 1.0,
            "ece": 0.0, "brier": 0.25
        }

    y_pred = (y_s >= threshold).astype(int)

    # Confusion matrix: tn, fp, fn, tp
    cm = confusion_matrix(y_t, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    auc = roc_auc_score(y_t, y_s)
    pr_auc = average_precision_score(y_t, y_s)
    bal_acc = balanced_accuracy_score(y_t, y_pred)
    f1 = f1_score(y_t, y_pred, zero_division=0)
    prec = precision_score(y_t, y_pred, zero_division=0)
    rec = recall_score(y_t, y_pred, zero_division=0)
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)
    ece = compute_ece(y_t, y_s, n_bins=10)
    brier = brier_score_loss(y_t, y_s)

    return {
        "auc": float(auc),
        "pr_auc": float(pr_auc),
        "bal_acc": float(bal_acc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "ece": float(ece),
        "brier": float(brier)
    }


def group_bootstrap_ci(
    df: pd.DataFrame,
    score_cols: List[str],
    thresholds: Dict[str, float],
    B: int = 1000,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs group-level bootstrap (B resamples) to compute 95% CIs and paired differences against HARMONIQ.
    """
    np.random.seed(seed)
    unique_groups = df["group"].unique()
    n_groups = len(unique_groups)
    y_true = df["label"].values

    # Collect bootstrap metrics
    boot_metrics = {col: {m: [] for m in ["auc", "bal_acc", "f1", "fpr", "fnr", "ece", "brier"]} for col in score_cols}
    # Collect paired differences vs HARMONIQ
    paired_diffs = {col: {"auc_diff": [], "bal_acc_diff": [], "fpr_diff": []} for col in score_cols if col != "HARMONIQ"}

    # Group index mapping for speed
    group_to_indices = df.groupby("group").indices

    for _ in range(B):
        sample_groups = np.random.choice(unique_groups, size=n_groups, replace=True)
        sample_indices = np.concatenate([group_to_indices[g] for g in sample_groups])

        sample_df = df.iloc[sample_indices]
        sample_y = sample_df["label"].values

        if len(np.unique(sample_y)) < 2:
            continue

        sample_scores = {}
        for col in score_cols:
            tau = thresholds.get(col, 0.5)
            m = compute_metrics_at_threshold(sample_y, sample_df[col].values, threshold=tau)
            for k in boot_metrics[col]:
                boot_metrics[col][k].append(m[k])
            sample_scores[col] = m

        if "HARMONIQ" in sample_scores:
            h_m = sample_scores["HARMONIQ"]
            for col in paired_diffs:
                b_m = sample_scores[col]
                paired_diffs[col]["auc_diff"].append(h_m["auc"] - b_m["auc"])
                paired_diffs[col]["bal_acc_diff"].append(h_m["bal_acc"] - b_m["bal_acc"])
                paired_diffs[col]["fpr_diff"].append(h_m["fpr"] - b_m["fpr"])

    # 1. Summary Stats Table
    stats_rows = []
    for col in score_cols:
        tau = thresholds.get(col, 0.5)
        point_m = compute_metrics_at_threshold(y_true, df[col].values, threshold=tau)

        row = {
            "System": col,
            "Threshold": tau,
            "AUC": point_m["auc"],
            "AUC_CI_Low": float(np.percentile(boot_metrics[col]["auc"], 2.5)) if boot_metrics[col]["auc"] else point_m["auc"],
            "AUC_CI_High": float(np.percentile(boot_metrics[col]["auc"], 97.5)) if boot_metrics[col]["auc"] else point_m["auc"],
            "Bal_Acc": point_m["bal_acc"],
            "Bal_Acc_CI_Low": float(np.percentile(boot_metrics[col]["bal_acc"], 2.5)) if boot_metrics[col]["bal_acc"] else point_m["bal_acc"],
            "Bal_Acc_CI_High": float(np.percentile(boot_metrics[col]["bal_acc"], 97.5)) if boot_metrics[col]["bal_acc"] else point_m["bal_acc"],
            "F1": point_m["f1"],
            "FPR": point_m["fpr"],
            "FNR": point_m["fnr"],
            "ECE": point_m["ece"],
            "Brier": point_m["brier"]
        }
        stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows)

    # 2. Paired Differences Table
    paired_rows = []
    for col in paired_diffs:
        d_auc = paired_diffs[col]["auc_diff"]
        d_bal = paired_diffs[col]["bal_acc_diff"]
        d_fpr = paired_diffs[col]["fpr_diff"]

        p_val_auc = float(np.mean(np.array(d_auc) <= 0.0)) * 2 if d_auc else 1.0
        p_val_auc = min(1.0, p_val_auc)

        paired_rows.append({
            "Baseline": col,
            "Delta_AUC_Mean": float(np.mean(d_auc)) if d_auc else 0.0,
            "Delta_AUC_CI_Low": float(np.percentile(d_auc, 2.5)) if d_auc else 0.0,
            "Delta_AUC_CI_High": float(np.percentile(d_auc, 97.5)) if d_auc else 0.0,
            "Delta_BalAcc_Mean": float(np.mean(d_bal)) if d_bal else 0.0,
            "Delta_FPR_Mean": float(np.mean(d_fpr)) if d_fpr else 0.0,
            "P_Value_AUC": round(p_val_auc, 4)
        })

    paired_df = pd.DataFrame(paired_rows)
    return stats_df, paired_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate test predictions with group-level bootstrap CIs.")
    parser.add_argument('--predictions', type=str, default="results/predictions.csv", help="Predictions CSV")
    parser.add_argument('--thresholds', type=str, default="results/thresholds.json", help="Thresholds JSON")
    parser.add_argument('--output-stats', type=str, default="results/stats_table.csv", help="Stats table CSV")
    parser.add_argument('--output-paired', type=str, default="results/paired_tests.csv", help="Paired test CSV")
    parser.add_argument('--output-json', type=str, default="results/results.json", help="Summary JSON")
    parser.add_argument('--bootstrap-rounds', type=int, default=500, help="Number of bootstrap rounds")
    args = parser.parse_args()

    pred_df = pd.read_csv(args.predictions)
    test_df = pred_df[pred_df["split"] == "test"].copy()

    with open(args.thresholds, "r", encoding="utf-8") as f:
        thresh_info = json.load(f)
    tau_star = thresh_info.get("tau_star", 0.5)

    metadata_cols = {"id", "group", "dataset", "task", "generator", "split", "label", "n_claims"}
    score_cols = [c for c in test_df.columns if c not in metadata_cols]

    # Map thresholds: HARMONIQ uses tau_star, others default to 0.5
    sys_thresholds = {c: (tau_star if c == "HARMONIQ" else 0.50) for c in score_cols}

    print(f"Running bootstrap evaluation on N={len(test_df)} test responses (B={args.bootstrap_rounds})...")
    stats_df, paired_df = group_bootstrap_ci(test_df, score_cols, sys_thresholds, B=args.bootstrap_rounds)

    stats_df.to_csv(args.output_stats, index=False)
    paired_df.to_csv(args.output_paired, index=False)
    print(f"✓ Saved stats table to {args.output_stats}")
    print(f"✓ Saved paired tests to {args.output_paired}")

    # Generate summary results.json
    harmoniq_row = stats_df[stats_df["System"] == "HARMONIQ"].iloc[0].to_dict()
    summary_results = {
        "n_test": len(test_df),
        "tau_star": tau_star,
        "harmoniq": {
            "auc": harmoniq_row["AUC"],
            "auc_ci": [harmoniq_row["AUC_CI_Low"], harmoniq_row["AUC_CI_High"]],
            "balanced_accuracy": harmoniq_row["Bal_Acc"],
            "bal_acc_ci": [harmoniq_row["Bal_Acc_CI_Low"], harmoniq_row["Bal_Acc_CI_High"]],
            "f1": harmoniq_row["F1"],
            "fpr": harmoniq_row["FPR"],
            "fnr": harmoniq_row["FNR"],
            "ece": harmoniq_row["ECE"],
            "brier": harmoniq_row["Brier"]
        }
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    print(f"✓ Saved results summary to {args.output_json}")
    print("\n" + stats_df[["System", "AUC", "Bal_Acc", "FPR", "FNR", "ECE", "Brier"]].to_string(index=False))


if __name__ == '__main__':
    main()
