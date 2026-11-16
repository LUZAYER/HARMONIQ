"""
Signal 4: Self-Verification and Stochastic Sample Consensus (V).

Evaluates cross-sample consensus across independently sampled completions Y^(1), ..., Y^(M).
Enforces:
- Returns NaN and available=False when stochastic samples do not exist (e.g., RAGTruth, HaluEval).
- Evaluates multi-sample consensus on benchmarks where samples exist (WikiBio-GPT3, FAVA).
- Designed for Adaptive Compute Gating (only invoked when fast signals indicate borderline risk).
"""

import time
import re
from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.signals.base import BaseSignal, SignalResult


STOPWORDS = {"the", "a", "an", "is", "was", "are", "were", "in", "on", "at", "to", "for", "of", "and", "or", "by", "with"}


class SelfVerificationSignal(BaseSignal):
    """Signal V: Stochastic multi-sample consensus verifier."""

    def __init__(
        self,
        version: str = "v1.0",
        cache_dir: str = "data/cache/signals",
        min_samples: int = 2
    ):
        super().__init__(name="selfverif", version=version, cache_dir=cache_dir)
        self.min_samples = min_samples

    def _compute(
        self,
        claim_text: str,
        evidence_text: str = "",
        query_text: str = "",
        samples: Optional[List[str]] = None
    ) -> SignalResult:
        t0 = time.time()
        claim_clean = claim_text.strip()

        # If samples are missing or fewer than min_samples, signal is unavailable (NaN)
        if not samples or len(samples) < self.min_samples or not claim_clean:
            return SignalResult(score=np.nan, available=False, latency_ms=0.0)

        valid_samples = [str(s).strip() for s in samples if str(s).strip()]
        if len(valid_samples) < self.min_samples:
            return SignalResult(score=np.nan, available=False, latency_ms=0.0)

        # Content-word overlap per sample (excluding stopwords)
        claim_words = set(re.findall(r'\b\w+\b', claim_clean.lower())) - STOPWORDS
        if not claim_words:
            claim_words = set(re.findall(r'\b\w+\b', claim_clean.lower()))
        if not claim_words:
            return SignalResult(score=np.nan, available=False, latency_ms=0.0)

        agreement_scores = []
        for s in valid_samples:
            s_words = set(re.findall(r'\b\w+\b', s.lower())) - STOPWORDS
            overlap = len(claim_words.intersection(s_words)) / len(claim_words)
            agreement_scores.append(overlap)

        # Mean consensus across stochastic completions
        mean_agreement = float(np.mean(agreement_scores))
        # Hallucination risk is high when cross-sample consensus is low
        hallucination_risk = float(np.clip(1.0 - mean_agreement, 0.0, 1.0))
        lat = (time.time() - t0) * 1000.0

        return SignalResult(
            score=hallucination_risk,
            available=True,
            latency_ms=lat,
            metadata={
                "num_samples": len(valid_samples),
                "mean_agreement": mean_agreement
            }
        )
