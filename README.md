# HARMONIQ: Calibrated Claim-Level Hybrid Hallucination Detection

[![Branch: rebuild](https://img.shields.io/badge/branch-rebuild-blue.svg)](#)
[![Reproducibility: 100%](https://img.shields.io/badge/reproducible-run__all.sh-green.svg)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-purple.svg)](#)

HARMONIQ is a claim-level hybrid hallucination detection framework that combines:
1. **Confidence Analysis ($C$):** Entity-targeted token log-probability entropy and linguistic hedging penalties.
2. **Retrieval Grounding ($R$):** Dense passage bi-encoder retrieval (BGE-small) with cross-encoder NLI contradiction scoring.
3. **Semantic Faithfulness ($S$):** Directional faithfulness verification paired with deterministic negation and date/number clash detection.
4. **Self-Verification ($V$):** Adaptive compute-gated multi-sample stochastic consensus.
5. **Length-Damped Aggregation (HCS):** Clausal length-damped soft-max pooling that resolves the compounding False Positive Rate dilemma across long responses.
6. **Native Missing-Value Branching:** HistGradientBoosting decision trees that natively handle missing/unretrieved evidence without synthetic imputation artifacts.

---

## 1. Quick Reproduction (Single Command)

To reproduce all dataset partitions, claim signals, trained trees, dev-tuned calibrations, evaluation metrics, statistical bootstrap tests, vector figures, and LaTeX macros from raw benchmark data:

```bash
# Clone and enter directory
git checkout rebuild

# Install dependencies in a virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the master reproduction pipeline
./run_all.sh
```

To compile the LaTeX research paper with all auto-generated macros:
```bash
pdflatex -interaction=nonstopmode HARMONIQ_paper.tex
pdflatex -interaction=nonstopmode HARMONIQ_paper.tex
```

---

## 2. Google Colab Execution (`HARMONIQ_framework.ipynb`)

For GPU-accelerated inference and large proxy LM scoring, run `HARMONIQ_framework.ipynb`:
1. Open [HARMONIQ_framework.ipynb](HARMONIQ_framework.ipynb) in Google Colab.
2. Select Runtime $\rightarrow$ **Change runtime type** $\rightarrow$ **T4 GPU**.
3. Execute **Runtime $\rightarrow$ Restart and Run All**.
4. The notebook runs top-to-bottom, writes `results/predictions.csv`, `results/claims.parquet`, `results/results.json`, and exports `results/numbers.tex`.

---

## 3. Repository Structure

```text
├── run_all.sh                           # Master one-command reproduction pipeline
├── HARMONIQ_paper.tex                   # IEEEtran paper (all numbers loaded via macros)
├── HARMONIQ_paper.pdf                   # Compiled publication-ready PDF
├── HARMONIQ_framework.ipynb             # Linear, self-contained Colab notebook
├── requirements.txt                     # Pinned dependencies
├── data/
│   ├── raw/                             # Raw benchmark sources (RAGTruth, HaluEval, TruthfulQA, WikiBio, FAVA)
│   ├── splits.json                      # Group-disjoint split mapping (seed 42)
│   └── cache/signals/                   # Parquet-cached claim signal scores
├── src/
│   ├── data/                            # Group-disjoint benchmark loaders (ragtruth, halueval, etc.)
│   ├── signals/                         # Signal stack (confidence, retrieval, semantic, selfverif)
│   ├── fusion/                          # HistGradientBoosting, isotonic calibration, HCS aggregators
│   └── baselines/                       # Supervised, MiniCheck, Vectara HHEM, AlignScore, SelfCheckGPT
├── experiments/
│   ├── make_splits.py                   # Generates data/splits.json and split_summary.csv
│   ├── score_all.py                     # Extracts claim signals into results/claims.parquet
│   ├── fit_fusion.py                    # Fits GBDT and exports results/predictions.csv
│   ├── evaluate.py                      # Group bootstrap CIs and paired difference tests
│   ├── ablate.py                        # Signal, aggregator, and imputation ablations
│   ├── fpr_vs_k.py                      # Multi-claim compounding risk study
│   ├── efficiency.py                    # Wall-clock latency and throughput benchmarks
│   ├── make_figures.py                  # Generates publication vector PDF/PNG figures
│   └── make_numbers.py                  # Exports results/numbers.tex macros
├── results/
│   ├── predictions.csv                  # Response-level evaluations across all systems
│   ├── claims.parquet                   # Claim-level extracted signals
│   ├── stats_table.csv                  # Main benchmark metrics with 95% CIs
│   ├── paired_tests.csv                 # Paired statistical differences vs. HARMONIQ
│   ├── ablation_results.csv             # Empirical ablation study results
│   ├── fpr_vs_k.csv                     # FPR vs. k data points
│   ├── efficiency_results.csv           # Latency and throughput breakdown
│   ├── numbers.tex                      # 240+ auto-generated LaTeX macros
│   └── figures/                         # Vector figures (ROC, PR, Calibration, FPR vs k, Workflow)
├── docs/
│   ├── claim_ledger.md                  # Line-by-line mapping of paper claims to data macros
│   └── unverified_refs.md               # Audit trail of removed unverified citations
└── tests/                               # Comprehensive unit test suite (loaders, signals, fusion, AST integrity)
```

---

## 4. Benchmark Datasets & Licenses

| Dataset | Primary Task | Partition Grouping | Official Reference | License |
|---|---|---|---|---|
| **RAGTruth** | QA, Summary, Data2txt | `source_id` | Niu et al. (ACL 2024) | MIT License |
| **HaluEval-QA** | Knowledge QA | `question` | Wang et al. (EMNLP 2023) | Apache 2.0 |
| **TruthfulQA** | Hallucination / Bias | `Question` | Lin et al. (ACL 2022) | Apache 2.0 |
| **WikiBio-GPT3** | Hallucinated Biographies | `wiki_bio_test_idx` | Manakul et al. (EMNLP 2023) | MIT License |
| **FAVA MultiSamples** | Fine-grained Detection | `prompt` | Mishra et al. (NAACL 2024) | Apache 2.0 |

*Strict Zero-Leakage Policy*: TruthfulQA is evaluated exclusively under retrieved Wikipedia passages (regime b) without access to the answer key (`Correct Answers`).

---

## 5. Non-Negotiable Scientific Integrity

1. **Zero Hardcoded Numbers:** No metric, AUC, latency, or curve is hand-typed into any script, table, or `.tex` file.
2. **AST-Enforced Figure Tests:** `tests/test_no_hardcoded_numbers.py` parses the AST of plotting scripts to prevent synthetic metric arrays.
3. **Paired Statistical Significance:** Superiority claims are accompanied by group-level paired bootstrap 95% confidence intervals and Holm-Bonferroni corrected $p$-values.
4. **Claim Ledger:** Every claim in the Abstract and Conclusion is documented in `docs/claim_ledger.md`.
