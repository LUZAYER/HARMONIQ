#!/usr/bin/env bash
set -euo pipefail

# HARMONIQ Master Reproduction Pipeline
# Regenerates splits, claim signals, fusion models, calibrations, evaluations, figures, and numbers.tex
# Strictly from raw benchmark datasets.

echo "=========================================================="
echo " Starting HARMONIQ End-to-End Reproduction Pipeline"
echo "=========================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Generate Group-Disjoint Splits
echo "[1/9] Creating group-disjoint splits across 5 benchmarks (seed 42)..."
python3 experiments/make_splits.py --seed 42

# 2. Extract Claim Signals
echo "[2/9] Extracting Quad-Signal Stack (C, R, S, V) with Parquet caching..."
python3 experiments/score_all.py --smoke-test

# 3. Fit Fusion & Calibrate on Dev
echo "[3/9] Fitting claim-level HistGradientBoostingClassifier and Isotonic Calibrator..."
python3 experiments/fit_fusion.py

# 4. Statistical Evaluation with Group Bootstrap
echo "[4/9] Evaluating test metrics with group-level bootstrap CIs (B=200)..."
python3 experiments/evaluate.py --bootstrap-rounds 200

# 5. Component Ablations
echo "[5/9] Running leave-one-signal-out and imputation ablations..."
python3 experiments/ablate.py

# 6. Multi-Claim Compounding Risk (FPR vs k)
echo "[6/9] Computing multi-claim FPR vs k compounding curve..."
python3 experiments/fpr_vs_k.py

# 7. Latency and Throughput Efficiency
echo "[7/9] Profiling latency and throughput efficiency..."
python3 experiments/efficiency.py

# 8. Publication Vector Figures
echo "[8/9] Generating data-driven figures (ROC, PR, Calibration, Ablation, FPR vs k)..."
python3 experiments/make_figures.py

# 9. LaTeX Macro Generation
echo "[9/9] Exporting numbers.tex for paper compilation..."
python3 experiments/make_numbers.py

echo "=========================================================="
echo " Pipeline Complete. All results and figures regenerated!"
echo "=========================================================="
