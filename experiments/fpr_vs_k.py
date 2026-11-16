#!/usr/bin/env python3
"""
Multi-Claim False Positive Rate (FPR vs. k) Analysis and Compounding Risk Study.

Investigates how response-level False Positive Rate escalates as claim count k grows:
- Empirical curves for HARMONIQ (HCS), Max Pooling, Mean Pooling, Noisy-OR
- Theoretical simulation under independence: FPR(k) = 1 - (1 - FPR_1)^k
- Binned by claim count: k in {1, 2-3, 4-6, 7+}

Outputs:
- results/fpr_vs_k.csv
- results/figures/fig_fpr_vs_k.pdf & png
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def assign_k_bin(k: int) -> str:
    if k == 1:
        return "1 claim"
    elif 2 <= k <= 3:
        return "2-3 claims"
    elif 4 <= k <= 6:
        return "4-6 claims"
    else:
        return "7+ claims"


def run_fpr_vs_k_analysis(
    predictions_csv: str = "results/predictions.csv",
    thresholds_json: str = "results/thresholds.json",
    output_csv: str = "results/fpr_vs_k.csv",
    output_fig_dir: str = "results/figures"
):
    df = pd.read_csv(predictions_csv)
    # Consider all factual responses (label == 0) across test (or test+dev if test is small)
    factual_df = df[df["label"] == 0].copy()

    with open(thresholds_json, "r", encoding="utf-8") as f:
        thresh = json.load(f)
    tau = thresh.get("tau_star", 0.50)

    factual_df["k_bin"] = factual_df["n_claims"].apply(assign_k_bin)
    bin_order = ["1 claim", "2-3 claims", "4-6 claims", "7+ claims"]

    systems = ["HARMONIQ", "Agg_Max", "Agg_Mean", "Agg_NoisyOR", "Agg_PowerMean_p2"]
    rows = []

    # 1. Compute empirical single-claim FPR (FPR_1)
    k1_df = factual_df[factual_df["k_bin"] == "1 claim"]
    fpr_1 = np.mean(k1_df["Agg_Max"].values >= tau) if len(k1_df) > 0 else 0.08
    fpr_1 = max(0.01, fpr_1)

    for b in bin_order:
        b_df = factual_df[factual_df["k_bin"] == b]
        n_resp = len(b_df)
        if n_resp == 0:
            continue

        row = {"k_bin": b, "N_Factual": n_resp}
        for sys_name in systems:
            if sys_name in b_df.columns:
                scores = b_df[sys_name].values
                # Threshold at tau
                fp_rate = np.mean(scores >= tau)
                row[sys_name] = round(float(fp_rate), 4)

        # Theoretical independent compounding curve
        # Typical representative k per bin: 1, 2.5, 5, 8
        k_rep = {"1 claim": 1, "2-3 claims": 2.5, "4-6 claims": 5, "7+ claims": 8}[b]
        row["Theoretical_Independent"] = round(float(1.0 - (1.0 - fpr_1) ** k_rep), 4)
        rows.append(row)

    res_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    res_df.to_csv(output_csv, index=False)
    print(f"✓ Saved FPR vs k table to {output_csv}")

    # Plot figure
    os.makedirs(output_fig_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=300)

    x = np.arange(len(res_df))
    ax.plot(x, res_df["Theoretical_Independent"] * 100, 'k--', lw=1.6, alpha=0.7, label='Independent Compounding Risk')

    palette = {
        "HARMONIQ": ("#b71c1c", 2.4, "-o"),
        "Agg_Max": ("#1565c0", 1.8, "-s"),
        "Agg_Mean": ("#2e7d32", 1.8, "-^"),
        "Agg_NoisyOR": ("#e65100", 1.8, "-d"),
        "Agg_PowerMean_p2": ("#6a1b9a", 1.8, "-x")
    }

    for sys_name in systems:
        if sys_name in res_df.columns:
            color, lw, marker = palette.get(sys_name, ("#333333", 1.5, "-o"))
            label_str = "HARMONIQ (HCS)" if sys_name == "HARMONIQ" else sys_name.replace("Agg_", "")
            ax.plot(x, res_df[sys_name] * 100, marker, color=color, lw=lw, label=label_str)

    ax.set_xticks(x)
    ax.set_xticklabels(res_df["k_bin"])
    ax.set_xlabel("Number of Claims per Response (k)")
    ax.set_ylabel("False Positive Rate on Factual Responses (%)")
    ax.set_title("Compounding Risk Dilemma: FPR vs. Claim Count (k)")
    ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#cccccc")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_ylim([0, 105])

    pdf_path = os.path.join(output_fig_dir, "fig_fpr_vs_k.pdf")
    png_path = os.path.join(output_fig_dir, "fig_fpr_vs_k.png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved FPR vs k plot to {pdf_path}")
    print("\n" + res_df.to_string(index=False))


if __name__ == '__main__':
    run_fpr_vs_k_analysis()
