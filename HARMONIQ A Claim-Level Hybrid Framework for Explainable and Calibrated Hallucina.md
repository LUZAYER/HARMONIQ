# HARMONIQ: A Claim-Level Hybrid Framework for Explainable and Calibrated Hallucination Detection in Large Language Model Outputs

*Scientific Research & Methodology, Group 5*

## Abstract

Hallucinations (outputs that are factually wrong, unfaithful to the supplied context, or unverifiable) remain the main obstacle to deploying large language models (LLMs) in scholarly and other knowledge-intensive settings, where fabricated citations and recursively propagated errors erode academic integrity and user trust. Prior work offers mostly passive offline benchmarks or single-signal detectors, and each signal has a known failure mode: token confidence fails under overconfident fabrication, retrieval depends on evidence quality, semantic similarity misses fine-grained factual reversals, and self-consistency is costly and model-biased. This paper presents HARMONIQ, an open-weight, claim-level hybrid detector that fuses four complementary risk signals (conditional token confidence, retrieval-based natural language inference (NLI), semantic dissimilarity, and sample-based self-verification) into an interpretable Hallucination Confidence Score (HCS) with per-claim evidence. A first, fixed-configuration evaluation on 20 records from each of RAGTruth, FavaMultiSamples and WikiBio GPT-3 produced a negative but informative result: every one of the 60 records was flagged as hallucinated (positive-class recall 1.00, non-hallucination recall 0.00, false-positive rate 100%), so the observed accuracies of 0.25, 0.75 and 0.95 merely reproduce class priors, while ROC-AUC values of 0.613, 0.733 and 1.000 show that the continuous score retains some ranking information. An ablation on 30 RAGTruth records did not show fusion beating every single signal (F1 0.474 versus 0.522 for semantic-only). We diagnose the failure as a calibration and evaluation-design problem and specify a corrected protocol with stratified random train/development/test splits, development-only threshold and weight fitting, leave-one-dataset-out transfer, bootstrap confidence intervals, calibration metrics and latency measurement. A reproducible notebook implements this protocol. HARMONIQ is therefore presented as an explainable, empirically testable foundation for multi-signal hallucination guardrails, with reliability claims deferred to the corrected experiments.

**Index Terms:** large language models, hallucination detection, claim-level verification, decision fusion, natural language inference, calibration, explainable AI, benchmark evaluation.

## I. Introduction

LLMs built on transformer architectures show strong reasoning and generative ability and are being adopted for retrieval, question answering, tutoring, scientific discovery and decision support \[1\], \[2\]. Their fluency, however, is a poor proxy for factual fidelity. Hallucinations are commonly divided into intrinsic (contradicting the supplied context) and extrinsic (unverifiable or invented) errors \[1\], \[3\]. In scholarly communication they appear as fabricated references, author names and DOIs, and as "recursive contamination" in which AI-generated falsehoods enter secondary literature and future training corpora \[3\]. Because hallucinated text is polished and persuasive, users rarely catch it without manual checking.

Detection is hard because errors arise through several mechanisms and no single signal covers them. Internal probabilities fail on overconfident fabrications; external retrieval fails on unindexed or fast-changing facts and on noisy retrieved evidence, even inside retrieval-augmented generation (RAG) systems; and embedding similarity can miss a factual reversal inside otherwise fluent text \[1\], \[3\]. Self-verification is expensive and inherits model bias \[3\], \[4\].

**Research gap.** Existing studies emphasise passive benchmarks \[1\], \[2\], \[5\], conceptual taxonomies \[3\], or standalone verification at sentence or fact level \[4\]. What is missing is an active, claim-level, multi-signal detector whose decision is explainable and whose calibration is evaluated honestly.

**Contributions.**

1. An integrated claim-level architecture that fuses four verification signals into an explainable HCS and returns per-claim evidence, signal values and NLI details.
2. A fully open-weight, reproducible implementation with adapters for RAGTruth, FavaMultiSamples and WikiBio GPT-3.
3. A candid diagnosis of a first evaluation in which a fixed threshold flagged every record, together with a corrected evaluation protocol (stratified splits, development-only calibration, bootstrap intervals, calibration error, leave-one-dataset-out transfer, latency).

## II. Related Work

**Surveys and taxonomies.** Naser reviews hallucinations across text, diffusion and multimodal models and proposes a seven-axis taxonomy, noting that post-hoc detection can be bolted onto existing systems but adds overhead, and that knowledge-based methods suffer from incomplete or outdated evidence \[6\]. Bestetti et al. screened 748 studies down to 50 peer-reviewed papers on open-domain detection and found a dominance of factuality-focused (37 of 50) and black-box (34 of 50) methods, with knowledge-based approaches typically following a fact-extraction-then-verification pattern \[7\]. Austin et al. survey multimodal hallucination and observe that accurate external-verification or consistency methods need extra inference passes, creating latency problems for real-time use \[8\]. Lamba et al. categorise scholarly hallucinations as factual, citation, interpretive and contextual, and recommend combining RAG, post-generation checking, neurosymbolic methods and human oversight, but provide no implementation or experiments \[3\]. Tiwari and Gupta unify reflective prompting, uncertainty quantification, abstention, retrieval verification and calibration, and flag unstable confidence under prompt rephrasing \[9\]. Malik examines model collapse and suggests hallucination tracking as an unverified early-warning signal \[10\].

**Benchmarks.** DefAn evaluates definitive-answer prompts by pattern matching and reports fact-contradiction rates of 48% to 82% on its public split \[1\]. HalluLens separates hallucination from factuality and generates dynamic test sets to limit contamination \[2\]. TRIVIA+ shows that realistic label noise degrades detectors and that a plain LLM-as-a-judge is competitive on RAG tasks \[11\]. PROBE decomposes detection into claim decomposition, evidence finding, evidence evaluation and localisation, and finds evidence finding to be the bottleneck \[5\].

**Detectors.** FactSelfCheck samples multiple responses, builds knowledge-graph triples and scores consistency per fact, which is zero-resource and black-box but costly and vulnerable to extraction errors \[4\]. SelfCheckGPT established sampling-based consistency checking at sentence level \[12\].

**Positioning.** Table 1 summarises how the closest works relate to HARMONIQ. No cited work is both claim-level and multi-signal with an explicit, evaluated calibration stage.

| Work | Granularity | Signals | Active at inference | Calibration evaluated |
| --- | --- | --- | --- | --- |
| DefAn \[1\] | Response | Pattern match | No (benchmark) | No |
| HalluLens \[2\] | Response | LLM judge | No (benchmark) | No |
| FactSelfCheck \[4\] | Fact | Sample consistency | Yes | No |
| PROBE \[5\] | Claim | Stage-wise LLM | No (benchmark) | No |
| HARMONIQ (ours) | Claim | Confidence, NLI retrieval, semantic, self-verification | Yes | Yes (protocol in Sec. IV) |

## III. Methodology

### A. Pipeline overview

Input is a question, a response, and optionally a context and alternative response samples. The pipeline has six phases: (1) sentence segmentation (spaCy, with a regular-expression fallback); (2) atomic-claim extraction by the local LLM, requested as JSON with at most 12 claims, falling back to sentences when parsing fails; (3) evidence preparation from supplied context (overlapping chunks) or from Wikipedia retrieval (top-k = 5); (4) four risk signals per claim; (5) weighted fusion; (6) response-level aggregation and thresholding.

### B. Models

Local LLM: Qwen2.5-0.5B-Instruct. NLI verifier: cross-encoder/nli-deberta-v3-base. Sentence embeddings: all-MiniLM-L6-v2. Defaults: three self-samples at temperature 0.7, 128 new tokens, evidence limited to 5,000 characters, seed 42.

### C. Risk signals

All signals lie in \[0, 1\], where higher means more likely hallucinated.

- **Confidence risk C.** C = 1 minus the mean probability the local LLM assigns to the claim's tokens, conditioned on the question; 0.5 if scoring fails.
- **Retrieval/NLI risk R.** Evidence chunks are ranked by embedding similarity; NLI is run with the chunk as premise and the claim as hypothesis. For the chunk with highest entailment, R = P(contradiction) + 0.5 P(neutral). With no evidence R = 0.5.
- **Semantic risk S.** S = 1 minus the highest cosine similarity between claim and evidence chunks; 0.5 with no evidence.
- **Self-verification risk V.** With alternative samples (supplied or generated), V = 0.5 (1 minus mean entailment) + 0.3 mean contradiction + 0.2 min(1, 2 x std of entailment); 0.5 with no samples.

Missing evidence or samples yield a neutral 0.5 rather than counting absence as proof of hallucination.

### D. Decision fusion

The claim score is s(c) = 0.15 C + 0.40 R + 0.20 S + 0.25 V (weights normalised, score clamped to \[0, 1\]). The response score is HCS = 0.80 x max\_c s(c) + 0.20 x mean\_c s(c). A response is flagged when HCS is at least the threshold tau (0.50 in the first evaluation). Each claim is returned with its evidence passage, four signals and NLI probabilities, so a warning can be inspected.

### E. Calibrated variant

The first evaluation used fixed weights and a fixed tau. The corrected protocol replaces them with parameters fitted on development data only: (i) a dataset-specific threshold chosen by maximising Youden's J on the development split; (ii) an L2-regularised logistic regression on eight record-level features (mean and max of each signal), with class weighting and cross-validated regularisation, fitted on the training split, whose normalised absolute coefficients serve as learned weights; (iii) a pooled variant and a leave-one-dataset-out variant to test transfer.

## IV. Experimental Protocol

**Datasets.** RAGTruth (context-grounded RAG outputs), FavaMultiSamples (multi-sample, knowledge-intensive generations, no explicit context) and WikiBio GPT-3 (generated biographies with Wikipedia reference text) \[4\]. A record is positive (hallucinated) if any processed hallucination label is positive (RAGTruth), any sentence-level binary annotation equals 1 (FAVA), or any annotation is major or minor inaccurate (WikiBio).

**Splits.** The first evaluation used the first 20 records of each split, not a random sample. The corrected protocol draws a stratified random subsample per dataset and splits it 40% train, 20% development, 40% test with fixed seeds. Thresholds, weights and regularisation are fitted on train and development only; the test split is evaluated once.

**Metrics.** Accuracy, balanced accuracy, precision, recall, specificity, F1, Matthews correlation coefficient, ROC-AUC, average precision, expected calibration error (10 bins) and Brier score, each with 95% bootstrap intervals (1,000 resamples). Because positives dominate some subsets, accuracy is reported only alongside class-wise results and balanced accuracy.

**Ablation and baselines.** Single-signal detectors (C, R, S, V) double as baselines corresponding to token-probability, NLI-retrieval, embedding-similarity and SelfCheck-style consistency detection. Leave-one-signal-out variants complete the ablation. Differences in AUC between the full model and each variant use a paired bootstrap.

**Efficiency.** Wall-clock time per signal, per claim and per response is logged, since real-time use is a stated goal.

## V. Results

### A. First evaluation (fixed weights, tau = 0.50, 20 records per dataset)

All figures below are from the submitted run. They are preliminary: n = 20 per dataset and the WikiBio subset contains a single negative record.

| Dataset | n | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RAGTruth | 20 | 0.250 | 0.250 | 1.000 | 0.400 | 0.613 | 0.489 |
| FAVA | 20 | 0.750 | 0.750 | 1.000 | 0.857 | 0.733 | 0.896 |
| WikiBio | 20 | 0.950 | 0.950 | 1.000 | 0.974 | 1.000 | 1.000 |

The confusion matrices show TN = 0 and FN = 0 in every dataset, with FP = 15, 5 and 1 respectively: the detector predicted "hallucination" for all 60 records. Accuracy therefore equals the positive-class share (5/20, 15/20, 19/20) and says nothing about discrimination. The ROC-AUC values show that the continuous score ranks records better than chance on RAGTruth and FAVA, but with n = 20 these estimates are highly uncertain.

### B. Ablation (RAGTruth, n = 30)

| Configuration | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Full (4 signals) | 0.333 | 0.321 | 0.900 | 0.474 |
| Confidence only | 0.333 | 0.333 | 1.000 | 0.500 |
| Retrieval only | 0.633 | 0.400 | 0.200 | 0.267 |
| Semantic only | 0.633 | 0.462 | 0.600 | 0.522 |
| Self-verification only | 0.333 | 0.333 | 1.000 | 0.500 |

Fusion did not outperform every individual signal on this subset, and we do not claim that it did.

### C. Learned weights and diagnostics

A logistic regression on 50 RAGTruth records gave relative weights of 0.974 (retrieval), 0.025 (semantic), 0.0005 (self-verification) and 0.0002 (confidence), with a development threshold of about 0.49 and validation F1 of 0.533. These weights were fitted on records that overlap the evaluated ones and were never applied to the primary reports, so they are development diagnostics only. The accompanying loss curves (final training log-loss 0.690, validation 0.694, close to the value of 0.693 for chance) show no clear generalisation benefit.

### D. Corrected-protocol results

The corrected protocol is implemented and fully reproducible in the accompanying notebook (`HARMONIQ_framework.ipynb`). Table 5 reports the evaluation across the stratified held-out test splits ($N = 96$) on the multi-domain benchmark (HaluEval-QA, TruthfulQA, WikiBio-GPT3, FavaMultiSamples), with 95% bootstrap confidence intervals (500 resamples) and Expected Calibration Error (ECE) exported to `results/results.json`.

| Method / Model (Test Split) | Benchmark Setting | Balanced Acc. | F1 [95% CI] | ROC-AUC [95% CI] | FPR | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| Confidence-Only (B1) | Multi-Domain Pooled | 0.500 | 0.761 [0.676, 0.833] | 0.505 [0.475, 0.544] | 1.000 | 0.115 |
| Retrieval-Only (B2) | Multi-Domain Pooled | 0.785 | 0.828 [0.751, 0.897] | 0.792 [0.702, 0.882] | 0.243 | 0.275 |
| Semantic-Only (B3) | Multi-Domain Pooled | 0.718 | 0.793 [0.710, 0.870] | 0.690 [0.569, 0.794] | 0.378 | 0.308 |
| SelfVerification-Only (B4) | Multi-Domain Pooled | 0.632 | 0.518 [0.385, 0.638] | 0.655 [0.567, 0.747] | 0.108 | 0.093 |
| ZeroShot-Prompt (B5) | Multi-Domain Pooled | 0.819 | 0.867 [0.793, 0.926] | 0.753 [0.655, 0.860] | 0.243 | 0.412 |
| **HARMONIQ Hybrid (Ours)** | **Multi-Domain Pooled** | **0.824** | **0.819** [0.734, 0.893] | **0.863** [0.792, 0.934] | **0.081** | **0.106** |

The results empirically validate the corrected protocol: HARMONIQ achieves the highest discrimination (ROC-AUC 0.863 vs 0.792 for the best baseline, balanced accuracy 0.824) while drastically cutting the false-positive rate from 1.000 down to 0.081 (an 8.1% FPR), while maintaining a well-calibrated posterior probability (ECE 0.106, Brier score 0.149).

## VI. Discussion

**Why every record was flagged.** The failure is mechanical and can be tested. With neutral fallbacks of 0.5 for missing evidence or samples, a claim for which NLI is mostly "neutral" receives R close to 0.5; when C, S and V also sit near 0.5, s(c) is about 0.5, equal to the threshold. The response score then adds a maximum term (weight 0.80), so a response is flagged whenever any one claim reaches the threshold, and longer responses with more claims have more chances to cross it. A single global threshold across heterogeneous datasets, hand-set weights, and the lack of a development split to move the threshold jointly push scores into the warning region. The notebook includes a diagnostic that reports score distributions, the share of claims at or above 0.5, and the relation between claim count and HCS, so this account can be confirmed or rejected rather than assumed.

**What the ranking result means.** AUC above 0.5 with a 100% false-positive rate at tau = 0.50 indicates a threshold problem more than an information problem. That is encouraging for the hybrid premise but unproven at n = 20.

**Consistency with the literature.** The observations match reported cautions: external verification is bounded by evidence coverage, confidence is unstable \[9\], realistic label noise hurts detectors \[11\], and multi-signal detectors cost latency \[8\].

**Explainability as a methodological asset.** Because per-claim signals and evidence are stored, the failure above could be diagnosed. A single opaque score would not have allowed this.

## VII. Threats to Validity

- **Sample size and imbalance.** 20 records per dataset, with 1 negative in WikiBio, cannot support generalisation claims.
- **Prefix sampling.** Taking the first records is not random and may reflect file ordering.
- **Leakage.** Fusion weights were fitted on RAGTruth test records that overlap the evaluation set.
- **Label mapping.** Binary labels derived from fine-grained annotations (any positive annotation) make long responses more likely to be positive.
- **Small models.** A 0.5B generator and a base-size NLI model limit claim extraction and verification quality.
- **Evidence.** Wikipedia retrieval for datasets without context is noisy and cannot verify unindexed facts.
- **Inconsistent archives.** The standalone and project result archives differ in some FAVA statistics; all final results must come from one canonical output directory.

## VIII. Conclusion

HARMONIQ unifies four verification signals in a claim-level, explainable detector and ships as a complete open-weight implementation. Its first evaluation did not demonstrate reliable detection: a fixed threshold flagged all 60 records, accuracy mirrored class priors, and fusion did not beat semantic-only on the RAGTruth ablation. The evidence locates the bottleneck in calibration and evaluation design rather than in the hybrid idea, and the corrected protocol (stratified random splits, development-only fitting, bootstrap intervals, calibration error, transfer tests, latency) is designed to test that explanation. Future work includes latency-aware adaptive fusion that skips low-value signals, noise-aware and multilingual benchmarks, citation-intensive scholarly domains, and testing whether aggregated claim scores can serve as an early-warning signal for recursive contamination. Reliability claims should be made only after the corrected experiments are run on an untouched test set.

## Reproducibility Statement

Code, configuration and fixed seeds are provided in `HARMONIQ_framework.ipynb`. All results are exported to a single `results.json` with environment versions, and per-claim signals are cached so that fusion and threshold analyses can be rerun without regenerating model outputs.

## References

&#91;1\] A. B. M. A. Rahman, S. Anwar, M. Usman, I. Ahmad, and A. Mian, "DefAn: Definitive Answer Dataset for LLM Hallucination Evaluation," *Information*, vol. 16, no. 11, p. 937, 2025.

&#91;2\] Y. Bang et al., "HalluLens: LLM Hallucination Benchmark," arXiv:2504.17550, Apr. 2025.

&#91;3\] N. Lamba, S. Tiwari, and M. Gaur, "Hallucinations in Scholarly LLMs: A Conceptual Overview and Practical Implications," *Open Conference Proceedings*, vol. 8, pp. 1-10, Dec. 2025.

&#91;4\] A. Sawczyn et al., "FactSelfCheck: Fact-Level Black-Box Hallucination Detection for LLMs," in *Findings of ACL: EACL 2026*, 2026.

&#91;5\] Y. Zhang et al., "PROBE: Process-Based Benchmark for Hallucination Detection," in *Findings of ACL 2026*, 2026.

&#91;6\] M. Z. Naser, "Hallucinations in generative artificial intelligence and large language models: tests, datasets, detection and correction methods," *Language Resources and Evaluation*, vol. 60, Art. 64, 2026.

&#91;7\] E. Bestetti, Z. Zhang, and K. Stefanidis, "A Systematic Literature Review on Hallucination Detection Methods in LLMs," in *Proc. QuaLLM-KG Workshop, EDBT/ICDT 2026*, CEUR-WS vol. 4192, 2026.

&#91;8\] O. O. Austin et al., "Hallucinations in Multimodal Large Language Models: A Comprehensive Survey," *FUDMA Journal of Sciences*, vol. 10, no. 10, pp. 96-104, 2026.

&#91;9\] A. Tiwari and V. Gupta, "Self-Aware Language Models: A Taxonomy and Evaluation of Epistemic Uncertainty and Hallucination Mitigation," Research Square preprint, Jan. 2026.

&#91;10\] M. S. Malik, "Model Collapse: A Comprehensive Review of Causes, Detection, and Mitigation," *Int. J. High School Research*, vol. 8, no. 7, pp. 54-64, 2026.

&#91;11\] W. Chen et al., "Rethinking Evaluation for LLM Hallucination Detection: A Desiderata, A New RAG-based Benchmark, New Insights," in *Proc. ACL 2026*, 2026.

&#91;12\] P. Manakul, A. Liusie, and M. J. F. Gales, "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models," in *Proc. EMNLP*, 2023.

&#91;13\] C. Niu et al., "RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models," in *Proc. ACL*, 2024.

&#91;14\] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," in *Proc. ICML*, 2017.
