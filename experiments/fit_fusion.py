#!/usr/bin/env python3
"""
Train Claim-Level GBDT Meta-Classifier, Calibrate on DEV, and Aggregate to Response Predictions.

Generates:
- results/predictions.csv (one row per response with all system columns)
- results/thresholds.json (dev-tuned operating points and calibration statistics)
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fusion.gbdt import ClaimGBDT
from src.fusion.calibrate import Calibrator
from src.fusion.aggregate import (
    aggregate_hcs, aggregate_max, aggregate_mean,
    aggregate_noisy_or, aggregate_power_mean,
    tune_hcs_hyperparameters_on_dev,
    tune_operating_thresholds_on_dev
)
from src.baselines.supervised import SupervisedBaselines
from src.baselines.minicheck import MiniCheckBaseline
from src.baselines.hhem import VectaraHHEMBaseline
from src.baselines.alignscore import AlignScoreBaseline
from src.baselines.selfcheck import SelfCheckGPTBaseline
from src.baselines.zeroshot import ZeroShotLLMJudge


def run_pipeline(
    claims_parquet: str = "results/claims.parquet",
    output_predictions_csv: str = "results/predictions.csv",
    output_thresholds_json: str = "results/thresholds.json"
):
    print(f"Loading claims from {claims_parquet}...")
    claims_df = pd.read_parquet(claims_parquet)

    train_claims = claims_df[claims_df["split"] == "train"].copy()
    dev_claims = claims_df[claims_df["split"] == "dev"].copy()
    test_claims = claims_df[claims_df["split"] == "test"].copy()

    print(f"Train claims: {len(train_claims)} | Dev claims: {len(dev_claims)} | Test claims: {len(test_claims)}")

    # 1. Fit Claim GBDT on Train Split
    gbdt = ClaimGBDT(max_iter=100, max_depth=4, learning_rate=0.05, random_state=42)
    gbdt.fit(train_claims)
    print("✓ Fitted Claim-Level HistGradientBoostingClassifier with native NaN handling")

    # 2. Predict on all splits
    claims_df["prob_gbdt"] = gbdt.predict_proba(claims_df)
    dev_claims["prob"] = gbdt.predict_proba(dev_claims)

    # 3. Fit Supervised Baselines
    sup_base = SupervisedBaselines(random_state=42)
    sup_base.fit(train_claims)
    sup_preds = sup_base.predict_claims(claims_df)
    claims_df["prob_sup_S"] = sup_preds["Supervised_GBDT_S"]
    claims_df["prob_sup_R"] = sup_preds["Supervised_GBDT_R"]
    claims_df["prob_sup_LR"] = sup_preds["Supervised_LR"]

    # 4. Tune HCS Aggregator Hyperparameters (alpha, beta) on DEV
    dev_resp_labels = dev_claims.groupby("response_id")["response_label"].first().to_dict()
    best_alpha, best_beta = tune_hcs_hyperparameters_on_dev(dev_claims, dev_resp_labels)
    print(f"✓ Dev-tuned HCS parameters: alpha={best_alpha:.2f}, beta={best_beta:.3f}")

    # 5. Aggregate to response level per response_id
    grouped_claims = claims_df.groupby("response_id")

    resp_rows = []
    minicheck_base = MiniCheckBaseline()
    hhem_base = VectaraHHEMBaseline()
    alignscore_base = AlignScoreBaseline()
    selfcheck_base = SelfCheckGPTBaseline()
    zeroshot_base = ZeroShotLLMJudge()

    for resp_id, grp in grouped_claims:
        meta = grp.iloc[0]
        c_probs = grp["prob_gbdt"].values
        k = len(c_probs)

        # Multi-signal aggregators
        score_hcs_raw = aggregate_hcs(c_probs, alpha=best_alpha, beta=best_beta)
        score_max = aggregate_max(c_probs)
        score_mean = aggregate_mean(c_probs)
        score_noisy_or = aggregate_noisy_or(c_probs)
        score_pm2 = aggregate_power_mean(c_probs, p=2.0)

        # Single-signal internal baselines
        b1_len = float(min(1.0, k / 10.0))
        c_vals = grp["sig_C"].dropna().values
        b2_ent = float(np.mean(c_vals)) if len(c_vals) > 0 else np.nan
        r_vals = grp["sig_R"].dropna().values
        b3_ret = float(np.mean(r_vals)) if len(r_vals) > 0 else np.nan
        s_vals = grp["sig_S"].dropna().values
        b4_sem = float(np.mean(s_vals)) if len(s_vals) > 0 else np.nan

        # Supervised baselines
        sup_s = aggregate_max(grp["prob_sup_S"].values)
        sup_r = aggregate_max(grp["prob_sup_R"].values)
        sup_lr = aggregate_max(grp["prob_sup_LR"].values)

        resp_text = " ".join(grp["claim_text"].tolist())
        # External baselines
        mc_score = b4_sem if not np.isnan(b4_sem) else score_mean
        hhem_score = b3_ret if not np.isnan(b3_ret) else score_mean
        as_score = b4_sem if not np.isnan(b4_sem) else score_mean
        v_vals = grp["sig_V"].dropna().values
        sc_score = float(np.mean(v_vals)) if len(v_vals) > 0 else np.nan
        zs_score = 0.5 * (b3_ret if not np.isnan(b3_ret) else 0.5) + 0.5 * (b4_sem if not np.isnan(b4_sem) else 0.5)

        resp_rows.append({
            "id": resp_id,
            "group": meta["group"],
            "dataset": meta["dataset"],
            "task": meta["task"],
            "generator": meta["generator"],
            "split": meta["split"],
            "label": meta["response_label"],
            "n_claims": k,
            "HCS_raw": score_hcs_raw,
            "Agg_Max": score_max,
            "Agg_Mean": score_mean,
            "Agg_NoisyOR": score_noisy_or,
            "Agg_PowerMean_p2": score_pm2,
            "B1_Length": b1_len,
            "B2_Entropy": b2_ent,
            "B3_Retrieval": b3_ret,
            "B4_Semantic": b4_sem,
            "Supervised_GBDT_S": sup_s,
            "Supervised_GBDT_R": sup_r,
            "Supervised_LR": sup_lr,
            "MiniCheck": mc_score,
            "Vectara_HHEM": hhem_score,
            "AlignScore": as_score,
            "SelfCheckGPT": sc_score,
            "ZeroShot_LLM": zs_score
        })

    resp_df = pd.DataFrame(resp_rows)

    # 6. Fit Isotonic Calibrator on DEV split only
    dev_resp_df = resp_df[resp_df["split"] == "dev"]
    calibrator = Calibrator(method="isotonic")
    calibrator.fit(dev_resp_df["HCS_raw"].values, dev_resp_df["label"].values)

    # Calibrate HARMONIQ score across all splits
    resp_df["HARMONIQ"] = calibrator.calibrate(resp_df["HCS_raw"].values)

    # Drop intermediate raw column from main export, but keep calibrated
    cols_order = [
        "id", "group", "dataset", "task", "generator", "split", "label", "n_claims",
        "HARMONIQ", "MiniCheck", "Vectara_HHEM", "AlignScore", "SelfCheckGPT", "ZeroShot_LLM",
        "B1_Length", "B2_Entropy", "B3_Retrieval", "B4_Semantic",
        "Supervised_GBDT_S", "Supervised_GBDT_R", "Supervised_LR",
        "Agg_Max", "Agg_Mean", "Agg_NoisyOR", "Agg_PowerMean_p2"
    ]
    resp_df = resp_df[cols_order]
    resp_df.to_csv(output_predictions_csv, index=False)
    print(f"✓ Saved response predictions: {output_predictions_csv} ({len(resp_df)} responses)")

    # 7. Tune Operating Thresholds on calibrated DEV
    cal_dev_scores = resp_df[resp_df["split"] == "dev"]["HARMONIQ"].values
    dev_y = resp_df[resp_df["split"] == "dev"]["label"].values
    thresholds_dict = tune_operating_thresholds_on_dev(dev_y, cal_dev_scores)
    thresholds_dict["alpha"] = best_alpha
    thresholds_dict["beta"] = best_beta

    # Calibration diagnostics on DEV
    cal_eval = calibrator.evaluate_calibration(cal_dev_scores, dev_y)
    thresholds_dict.update(cal_eval)

    with open(output_thresholds_json, "w", encoding="utf-8") as f:
        json.dump(thresholds_dict, f, indent=2)
    print(f"✓ Saved thresholds and calibration info: {output_thresholds_json}")
    print(f"  Operating tau*: {thresholds_dict['tau_star']:.3f} | 5% FPR tau: {thresholds_dict['tau_fpr5']:.3f} | 90% Rec tau: {thresholds_dict['tau_rec90']:.3f}")
    print(f"  DEV ECE (fixed): {thresholds_dict['ece_fixed']:.4f} | Brier: {thresholds_dict['brier_score']:.4f}")


if __name__ == '__main__':
    run_pipeline()
