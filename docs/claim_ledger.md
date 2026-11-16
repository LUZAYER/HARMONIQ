# HARMONIQ Scientific Claim Ledger

This ledger links every empirical claim made in the **Abstract** and **Conclusion** of `HARMONIQ_paper.tex` directly to its supporting LaTeX macro in `results/numbers.tex`, data row in `results/stats_table.csv`, or empirical ablation table.

Integrity Rule:
> "Every superiority claim needs a paired, group-level bootstrap CI that excludes zero, corrected for multiple comparisons. Otherwise write comparable or not significantly different."

---

## 1. Abstract Claims

| Abstract Sentence | Supporting Macro / File | Empirical Evidence & Ground Truth |
|---|---|---|
| **Sentence 1 (Problem)**: *"Large Language Models (LLMs) deployed in critical domains frequently generate hallucinations that conventional sequence-level and single-signal detectors fail to reliably localize or calibrate."* | Section I Background; Manakul et al. (2023), Niu et al. (2024). | Well-established NLP literature finding; motivated by high false alarm rates in multi-claim responses. |
| **Sentence 2 (Method)**: *"We present HARMONIQ, a claim-level hybrid hallucination detection framework that combines entity-targeted proxy-LM confidence ($C$), dense retrieval NLI ($R$), directional semantic faithfulness ($S$), and adaptive self-verification ($V$) via a gradient-boosted meta-classifier with native missing-value branching."* | Code: `src/fusion/gbdt.py`, `src/signals/` | Framework architectural definition; native NaN handling verified by unit tests in `tests/test_fusion_and_aggregation.py`. |
| **Sentence 3 (Multi-Claim Finding)**: *"Crucially, we investigate the multi-claim compounding risk dilemma, demonstrating empirically that uncalibrated max pooling causes false positive rates to inflate as response length $k$ grows, whereas length-damped soft-max aggregation (HCS) mitigates this degradation."* | Table IV; Figure 7; `results/fpr_vs_k.csv` | Empirical FPR vs $k$ analysis: Max pooling FPR escalates from 0% ($k=1$) to 23.5% ($k \ge 7$), whereas Noisy-OR reaches 100% FPR. HCS mitigates false alarm inflation. |
| **Sentence 4 (Benchmark Coverage)**: *"Evaluated on a unified corpus of $\Datasettotalresponses$ responses across five benchmarks (RAGTruth, HaluEval-QA, TruthfulQA, WikiBio-GPT3, and FAVA), HARMONIQ achieves an overall ROC-AUC of $\Aucharmoniq$ [95\% CI: $\Auclowharmoniq$, $\Auchighharmoniq$] and a balanced accuracy of $\Balaccharmoniq$."* | `\Datasettotalresponses`, `\Aucharmoniq`, `\Auclowharmoniq`, `\Auchighharmoniq`, `\Balaccharmoniq` in `results/numbers.tex` | Data: `results/split_summary.csv` and `results/stats_table.csv`. Exact point estimates and group-bootstrap 95% CIs. |
| **Sentence 5 (Baseline Comparison)**: *"HARMONIQ significantly outperforms zero-shot LLM judges ($\Delta \text{AUC} = +0.166$, $p < 0.001$) and linear feature fusion ($\Delta \text{AUC} = +0.104$, $p < 0.001$), while performing competitively with single-signal neural verifiers like MiniCheck ($\Delta \text{AUC} = +0.016$, 95\% CI: $[-0.024, +0.071]$)."* | `results/paired_tests.csv` | Paired group bootstrap tests: Zero-Shot LLM difference excludes 0 ($p < 0.001$); Supervised LR difference excludes 0 ($p < 0.001$); MiniCheck CI contains zero, reported objectively as competitive without claiming superiority. |
| **Sentence 6 (Calibration & Latency)**: *"Following dev-tuned isotonic calibration, HARMONIQ achieves an expected calibration error (ECE) of $\Eceharmoniq$ and a Brier score of $\Brierharmoniq$, operating with an end-to-end throughput of $\LatencyMeanEndToEndPipeline$ ms per response."* | `\Eceharmoniq`, `\Brierharmoniq`, `\LatencyMeanEndToEndPipeline` in `results/numbers.tex` | Data: `results/thresholds.json` and `results/efficiency_results.csv`. |

---

## 2. Conclusion Claims

| Conclusion Claim | Supporting Macro / File | Empirical Ground Truth |
|---|---|---|
| **Claim 1 (Multi-Signal Synergy)**: *"Ablation experiments confirm that removing self-verification ($V$) incurs a loss of $0.047$ AUC, while native NaN handling outperforms synthetic mean imputation by $+0.011$ AUC."* | `results/ablation_results.csv`, `\Ablaucwoselfverificationv` | Empirical ablation table: Full HARMONIQ AUC 0.816 vs w/o V 0.769 ($\Delta = -0.047$); Native NaN (0.816) vs Mean Imputation (0.805, $\Delta = -0.011$). |
| **Claim 2 (Aggregation Mechanics)**: *"Length-damped soft-max pooling (HCS) prevents the catastrophic false alarm escalation inherent in independence-assuming aggregators like Noisy-OR."* | `results/fpr_vs_k.csv`, Figure 7 | On responses with $k \ge 7$ claims, Noisy-OR suffers 100.0% false positive rate on factual answers, while HCS maintains controlled FPR. |
| **Claim 3 (Reproducibility)**: *"All splits, signal extractions, trained trees, evaluations, and figures are fully reproduced via `run_all.sh` from raw benchmark sources."* | Code: `run_all.sh` | Deterministic end-to-end execution verified. |
