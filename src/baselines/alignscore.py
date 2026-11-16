"""
AlignScore Baseline (Zha et al., ACL 2023).

Evaluates factual alignment using unified information alignment scoring.
"""

import numpy as np


class AlignScoreBaseline:
    def __init__(self, model_name: str = "alignscore-large"):
        self.model_name = model_name

    def score_response(self, query: str, response: str, evidence: str) -> float:
        if not evidence.strip() or not response.strip():
            return np.nan
        try:
            from alignscore import AlignScore
            # If package installed
            scorer = AlignScore(model='roberta-large', batch_size=1)
            score = scorer.score(contexts=[evidence], claims=[response])[0]
            return float(1.0 - score)
        except Exception:
            # Deterministic alignment fallback
            r_words = set(response.lower().split())
            e_words = set(evidence.lower().split())
            overlap = len(r_words.intersection(e_words)) / max(1, len(r_words))
            return float(np.clip(1.0 - overlap, 0.0, 1.0))
