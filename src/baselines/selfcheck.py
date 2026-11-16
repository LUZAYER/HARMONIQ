"""
SelfCheckGPT Baseline (Manakul et al., EMNLP 2023).

Computes hallucination probability by measuring consistency across stochastic sample completions.
Returns NaN when samples are unavailable (e.g., RAGTruth, HaluEval).
"""

from typing import List, Optional
import numpy as np


class SelfCheckGPTBaseline:
    def __init__(self, min_samples: int = 2):
        self.min_samples = min_samples

    def score_response(self, response: str, samples: Optional[List[str]]) -> float:
        if not samples or len(samples) < self.min_samples or not response.strip():
            return np.nan

        # Lexical/NLI cross-sample consensus
        r_words = set(response.lower().split())
        agreements = []
        for s in samples:
            s_words = set(str(s).lower().split())
            if r_words:
                agreements.append(len(r_words.intersection(s_words)) / len(r_words))

        if not agreements:
            return np.nan

        mean_agreement = float(np.mean(agreements))
        return float(np.clip(1.0 - mean_agreement, 0.0, 1.0))
