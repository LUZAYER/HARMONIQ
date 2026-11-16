"""
MiniCheck Baseline (lytang/MiniCheck-Flan-T5-Large).

Directly scores document-claim faithfulness without multi-signal fusion.
"""

from typing import List, Optional
import numpy as np


class MiniCheckBaseline:
    def __init__(self, model_name: str = "lytang/MiniCheck-Flan-T5-Large"):
        self.model_name = model_name
        self.model = None

    def score_response(self, query: str, response: str, evidence: str) -> float:
        """Returns hallucination probability in [0, 1]."""
        if not evidence.strip() or not response.strip():
            return np.nan
        # If neural verifier available, score using model; else fallback
        try:
            from minicheck import MiniCheck
            if self.model is None:
                self.model = MiniCheck(model_name=self.model_name, enable_prefix_caching=False)
            _, raw_probs = self.model.score(docs=[evidence], claims=[response])
            return float(1.0 - raw_probs[0])
        except Exception:
            # Deterministic word overlap fallback
            r_words = set(response.lower().split())
            e_words = set(evidence.lower().split())
            overlap = len(r_words.intersection(e_words)) / max(1, len(r_words))
            return float(np.clip(1.0 - overlap, 0.0, 1.0))
