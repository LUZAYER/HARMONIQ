"""
Baseline Hallucination Detection Systems.

Includes:
- B1 to B4: Internal single-signal baselines (Length, Entropy/Confidence, Retrieval, Semantic)
- Supervised single-signal baselines (GBDT-on-MiniCheck, GBDT-on-NLI, Logistic Regression)
- MiniCheck-Flan-T5-Large
- Vectara HHEM-2.1-Open
- AlignScore
- SelfCheckGPT (NLI consensus)
- ZeroShot LLM Judge
"""

from src.baselines.supervised import SupervisedBaselines
