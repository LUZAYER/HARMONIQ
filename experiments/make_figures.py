#!/usr/bin/env python3
"""
Dynamic Vector Graphics and Figure Generator for HARMONIQ Research Paper.

Generates publication-quality 300+ DPI vector PDF and PNG figures strictly
from evaluated predictions (results/predictions.csv) and empirical ablation
tables (results/ablation_results.csv).

Integrity Guarantee:
Zero hard-coded curves, metric values, or hand-typed arrays.
All curves and metrics are computed dynamically using scikit-learn.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve


# Setup publication typography and styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'mathtext.fontset': 'stix'
})

# Consistent palette for systems
SYSTEM_COLORS = {
    'HARMONIQ': '#b71c1c',       # Crimson
    'HARMONIQ (Ours)': '#b71c1c',
    'MiniCheck': '#1565c0',      # Dark Blue
    'MiniCheck-7B': '#1565c0',
    'Vectara_HHEM': '#2e7d32',   # Green
    'Vectara HHEM-2.1': '#2e7d32',
    'AlignScore': '#e65100',     # Amber / Orange
    'SelfCheckGPT': '#6a1b9a',   # Purple
    'ZeroShot_LLM': '#00838f',   # Teal
    'B1_Length': '#757575',      # Grey
    'B2_Entropy': '#546e7a',     # Slate
    'B3_Retrieval': '#37474f',   # Charcoal
    'B4_Semantic': '#455a64'
}


def draw_workflow_diagram(output_dir: str):
    """
    Generate the architectural schematic workflow diagram (fig_workflow.pdf/png).
    This diagram visualizes pipeline structure and dataflow without empirical metric curves.
    """
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=300)
    ax.axis('off')

    c_input = '#e3f2fd'
    c_decomp = '#e0f2f1'
    c_signals = '#fff8e1'
    c_fusion = '#f3e5f5'
    c_output = '#e8f5e9'
    c_border = '#263238'

    def draw_card(xy, w, h, header, sublines, bg_color):
        card = patches.FancyBboxPatch(
            xy, w, h, boxstyle='round,pad=0.015,rounding_size=0.025',
            fc=bg_color, ec=c_border, lw=1.4
        )
        ax.add_patch(card)
        cx = xy[0] + w / 2.0
        top_y = xy[1] + h - 0.05
        ax.text(cx, top_y, header, ha='center', va='top', fontsize=9.5, fontweight='bold', color='#1a237e')
        body_text = '\n'.join(sublines)
        ax.text(cx, xy[1] + (h - 0.06) / 2.0, body_text, ha='center', va='center', fontsize=7.8, color='#263238', linespacing=1.25)

    # 1. Input Box
    draw_card((0.01, 0.38), 0.16, 0.28, "1. Input Query & Resp",
              ["User Query (Q)", "+", "LLM Generation (Y)", "Grounding Passages (E)"], c_input)

    # 2. Decomposer Box
    draw_card((0.21, 0.38), 0.18, 0.28, "2. Claim Decomposer",
              ["Clausal Decomposition", "Atomic Claims {c_1, ..., c_k}", "Character Spans [S_j, E_j]", "Entities E(c_j) & N(c_j)"], c_decomp)

    # 3. Quad Signal Stack
    draw_card((0.43, 0.76), 0.26, 0.21, "Confidence (C)",
              ["Entity-Targeted Token Entropy", "Proxy LM Teacher-Forced P(t)", "Hedge Linguistic Penalty"], c_signals)
    draw_card((0.43, 0.51), 0.26, 0.21, "Retrieval NLI (R)",
              ["Dense Bi-Encoder (BGE-small)", "DeBERTa-v3 Cross-Encoder", "P(contradiction) + 0.5 P(neutral)"], c_signals)
    draw_card((0.43, 0.26), 0.26, 0.21, "Semantic Alignment (S)",
              ["Directional Faithfulness Verifier", "Numeric / Date Discrepancy", "Negation Polarity Check"], c_signals)
    draw_card((0.43, 0.01), 0.26, 0.21, "Self-Verification (V)",
              ["Adaptive Compute Gating", "Cross-Sample Consensus", "NaN on Zero-Sample Tasks"], c_signals)

    # 4. Fusion Engine
    draw_card((0.73, 0.40), 0.26, 0.26, "4. Calibrated Fusion",
              ["Claim GBDT (HistGradientBoosting)", "Native Missing-Value Splits", "Length-Damped Aggregation (HCS)", "Isotonic Calibrator (Dev-Tuned)"], c_fusion)

    # 5. Output Verification Report
    draw_card((0.73, 0.05), 0.26, 0.26, "5. Verification Audit",
              ["Calibrated Score HCS in [0, 1]", "Localized Claim Spans", "Three-Tier Risk Badging", "Attribution Diagnostic Report"], c_output)

    # Connecting Arrows
    arr = dict(arrowstyle='->', lw=1.6, color='#263238')
    ax.annotate('', xy=(0.21, 0.52), xytext=(0.17, 0.52), arrowprops=arr)

    # Decomp to Signals
    ax.annotate('', xy=(0.43, 0.86), xytext=(0.39, 0.56), arrowprops=arr)
    ax.annotate('', xy=(0.43, 0.61), xytext=(0.39, 0.52), arrowprops=arr)
    ax.annotate('', xy=(0.43, 0.36), xytext=(0.39, 0.48), arrowprops=arr)
    ax.annotate('', xy=(0.43, 0.11), xytext=(0.39, 0.44), arrowprops=arr)

    # Signals to Fusion
    ax.annotate('', xy=(0.73, 0.56), xytext=(0.69, 0.86), arrowprops=arr)
    ax.annotate('', xy=(0.73, 0.53), xytext=(0.69, 0.61), arrowprops=arr)
    ax.annotate('', xy=(0.73, 0.50), xytext=(0.69, 0.36), arrowprops=arr)
    ax.annotate('', xy=(0.73, 0.47), xytext=(0.69, 0.11), arrowprops=arr)

    # Fusion to Output
    ax.annotate('', xy=(0.86, 0.31), xytext=(0.86, 0.40), arrowprops=arr)

    plt.tight_layout()
    pdf_path = os.path.join(output_dir, "fig_workflow.pdf")
    png_path = os.path.join(output_dir, "fig_workflow.png")
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    fig.savefig(png_path, format='png', bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved workflow diagram: {pdf_path}")


def plot_roc_curves(df: pd.DataFrame, score_cols: list, label_col: str, output_dir: str):
    """
    Compute and plot real ROC curves dynamically from predictions.
    """
    fig, ax = plt.subplots(figsize=(5.4, 4.2), dpi=300)
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.6, label='Random Chance (AUC = 0.500)')

    y_true = df[label_col].values

    for col in score_cols:
        if col not in df.columns:
            continue
        scores = df[col].values
        valid_mask = ~np.isnan(scores)
        if valid_mask.sum() == 0:
            continue

        fpr, tpr, _ = roc_curve(y_true[valid_mask], scores[valid_mask])
        score_auc = auc(fpr, tpr)
        color = SYSTEM_COLORS.get(col, None)
        lw = 2.2 if 'harmoniq' in col.lower() else 1.5
        ax.plot(fpr, tpr, lw=lw, color=color, label=f"{col} (AUC = {score_auc:.3f})")

    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)')
    ax.set_title('Receiver Operating Characteristic (ROC) Comparison')
    ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='#cccccc')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    pdf_path = os.path.join(output_dir, "fig1_roc_curves.pdf")
    png_path = os.path.join(output_dir, "fig1_roc_curves.png")
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    fig.savefig(png_path, format='png', bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved ROC curves: {pdf_path}")


def plot_pr_curves(df: pd.DataFrame, score_cols: list, label_col: str, output_dir: str):
    """
    Compute and plot real Precision-Recall curves dynamically from predictions.
    """
    fig, ax = plt.subplots(figsize=(5.4, 4.2), dpi=300)
    y_true = df[label_col].values
    baseline_prev = np.mean(y_true)
    ax.axhline(y=baseline_prev, color='k', linestyle='--', lw=1.2, alpha=0.6,
               label=f'Prevalence Base ({baseline_prev:.3f})')

    for col in score_cols:
        if col not in df.columns:
            continue
        scores = df[col].values
        valid_mask = ~np.isnan(scores)
        if valid_mask.sum() == 0:
            continue

        prec, rec, _ = precision_recall_curve(y_true[valid_mask], scores[valid_mask])
        ap = average_precision_score(y_true[valid_mask], scores[valid_mask])
        color = SYSTEM_COLORS.get(col, None)
        lw = 2.2 if 'harmoniq' in col.lower() else 1.5
        ax.plot(rec, prec, lw=lw, color=color, label=f"{col} (PR-AUC = {ap:.3f})")

    ax.set_xlabel('Recall (True Positive Rate)')
    ax.set_ylabel('Precision (Positive Predictive Value)')
    ax.set_title('Precision-Recall (PR) Curves')
    ax.legend(loc='lower left', frameon=True, fancybox=False, edgecolor='#cccccc')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0.0, 1.02])

    pdf_path = os.path.join(output_dir, "fig2_pr_curves.pdf")
    png_path = os.path.join(output_dir, "fig2_pr_curves.png")
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    fig.savefig(png_path, format='png', bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved PR curves: {pdf_path}")


def plot_calibration_curves(df: pd.DataFrame, score_cols: list, label_col: str, output_dir: str):
    """
    Compute and plot real reliability/calibration curves dynamically from predictions.
    """
    fig, ax = plt.subplots(figsize=(5.4, 4.2), dpi=300)
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.6, label='Perfect Calibration')

    y_true = df[label_col].values

    for col in score_cols:
        if col not in df.columns:
            continue
        scores = df[col].values
        valid_mask = ~np.isnan(scores)
        if valid_mask.sum() == 0:
            continue

        prob_true, prob_pred = calibration_curve(y_true[valid_mask], scores[valid_mask], n_bins=10, strategy='uniform')
        brier = brier_score_loss(y_true[valid_mask], scores[valid_mask])
        color = SYSTEM_COLORS.get(col, None)
        lw = 2.0 if 'harmoniq' in col.lower() else 1.4
        ax.plot(prob_pred, prob_true, marker='o', markersize=4, lw=lw, color=color,
                label=f"{col} (Brier = {brier:.3f})")

    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Empirical Fraction of Positives')
    ax.set_title('Calibration Reliability Diagram')
    ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='#cccccc')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    pdf_path = os.path.join(output_dir, "fig3_calibration_reliability.pdf")
    png_path = os.path.join(output_dir, "fig3_calibration_reliability.png")
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    fig.savefig(png_path, format='png', bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved calibration curves: {pdf_path}")


def plot_ablation_comparison(ablation_csv: str, output_dir: str):
    """
    Plot empirical ablation comparison from results/ablation_results.csv.
    """
    if not os.path.exists(ablation_csv):
        print(f"Ablation file {ablation_csv} not found, skipping ablation plot.")
        return

    df = pd.read_csv(ablation_csv)
    fig, ax1 = plt.subplots(figsize=(6.0, 4.0), dpi=300)

    # Columns: Component/Configuration, ROC-AUC (or AUC), Balanced Accuracy (or Bal_Acc), FPR
    configs = df.iloc[:, 0].astype(str).tolist()
    x = np.arange(len(configs))
    width = 0.25

    auc_col = [c for c in df.columns if 'auc' in c.lower()][0]
    bal_acc_col = [c for c in df.columns if 'bal' in c.lower() or 'acc' in c.lower()][0]
    fpr_col = [c for c in df.columns if 'fpr' in c.lower()][0]

    rects1 = ax1.bar(x - width, df[auc_col].values, width, label='ROC-AUC', color='#1565c0', edgecolor='black', lw=0.6)
    rects2 = ax1.bar(x, df[bal_acc_col].values, width, label='Balanced Acc.', color='#2e7d32', edgecolor='black', lw=0.6)

    ax2 = ax1.twinx()
    rects3 = ax2.bar(x + width, df[fpr_col].values * 100, width, label='False Positive Rate (%)', color='#e65100', edgecolor='black', lw=0.6)

    ax1.set_ylabel('Metric Value')
    ax2.set_ylabel('FPR (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=25, ha='right')
    ax1.set_ylim([0.4, 1.0])
    ax2.set_ylim([0, 30])
    ax1.set_title('Ablation Study: Impact of Components')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', frameon=True, fancybox=False, edgecolor='#cccccc')

    pdf_path = os.path.join(output_dir, "fig5_ablation_comparison.pdf")
    png_path = os.path.join(output_dir, "fig5_ablation_comparison.png")
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    fig.savefig(png_path, format='png', bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved ablation comparison: {pdf_path}")


def plot_confusion_matrix(df: pd.DataFrame, score_col: str, label_col: str, threshold: float, output_dir: str):
    """
    Compute and plot real confusion matrix dynamically from predictions at the calibrated threshold.
    """
    if score_col not in df.columns or label_col not in df.columns:
        return

    y_true = df[label_col].values
    y_scores = df[score_col].values
    valid_mask = ~np.isnan(y_scores)
    y_pred = (y_scores[valid_mask] >= threshold).astype(int)

    cm = confusion_matrix(y_true[valid_mask], y_pred)

    fig, ax = plt.subplots(figsize=(4.5, 3.8), dpi=300)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = ['Factual (0)', 'Hallucinated (1)']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=f'Confusion Matrix ({score_col} @ tau={threshold:.2f})',
           ylabel='Ground Truth Label',
           xlabel='Predicted Label')

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight='bold')

    pdf_path = os.path.join(output_dir, "fig6_confusion_matrix.pdf")
    png_path = os.path.join(output_dir, "fig6_confusion_matrix.png")
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    fig.savefig(png_path, format='png', bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved confusion matrix: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures from empirical predictions.")
    parser.add_argument('--predictions', type=str, default="results/predictions.csv",
                        help="Path to results/predictions.csv")
    parser.add_argument('--ablation', type=str, default="results/ablation_results.csv",
                        help="Path to results/ablation_results.csv")
    parser.add_argument('--output-dir', type=str, default="results/figures",
                        help="Directory to save output figures")
    parser.add_argument('--threshold', type=float, default=0.50,
                        help="Operating classification threshold")
    parser.add_argument('--workflow-only', action='store_true',
                        help="Generate only the architectural workflow diagram")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Workflow Diagram (Independent of data)
    draw_workflow_diagram(args.output_dir)
    if args.workflow_only:
        return

    # 2. Ablation plot
    if os.path.exists(args.ablation):
        plot_ablation_comparison(args.ablation, args.output_dir)

    # 3. Prediction-dependent curves
    if not os.path.exists(args.predictions):
        print(f"Notice: Predictions file '{args.predictions}' does not exist yet.")
        print("Figures requiring prediction evaluations will be generated when predictions.csv is produced.")
        return

    df = pd.read_csv(args.predictions)
    label_col = 'label'
    if label_col not in df.columns:
        print(f"Error: label column '{label_col}' not found in {args.predictions}.")
        return

    # Identify score columns (exclude metadata columns)
    metadata_cols = {'id', 'group', 'dataset', 'task', 'generator', 'split', 'label', 'n_claims'}
    score_cols = [c for c in df.columns if c not in metadata_cols]

    if score_cols:
        plot_roc_curves(df, score_cols, label_col, args.output_dir)
        plot_pr_curves(df, score_cols, label_col, args.output_dir)
        plot_calibration_curves(df, score_cols, label_col, args.output_dir)
        primary_col = 'HARMONIQ' if 'HARMONIQ' in score_cols else score_cols[0]
        plot_confusion_matrix(df, primary_col, label_col, args.threshold, args.output_dir)


if __name__ == '__main__':
    main()
