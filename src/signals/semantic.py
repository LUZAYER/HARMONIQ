"""
Signal 3: Semantic Alignment and Directional Faithfulness (S).

Combines:
1. Directional Faithfulness Verifier (MiniCheck-Flan-T5-Large or NLI directional probability)
2. Deterministic Negation Polarity Check
3. Deterministic Numerical and Date Discrepancy Verification

Formula:
    S(c_j, E) in [0.0, 1.0], where 1.0 indicates hallucination.
    If evidence is missing, available = False and score = NaN.
"""

import re
import time
from typing import List, Optional, Set
import numpy as np

from src.signals.base import BaseSignal, SignalResult


NEGATION_WORDS = {"not", "never", "no", "neither", "nor", "none", "without", "hardly", "barely", "failed", "denied", "refused"}
YEAR_PATTERN = re.compile(r'\b(19\d\d|20\d\d)\b')
NUMBER_PATTERN = re.compile(r'\b\d+(?:[\.,]\d+)?\b')


def detect_negation_clash(claim: str, evidence: str) -> bool:
    """Detects explicit negation polarity mismatch between claim and evidence."""
    claim_tokens = set(re.findall(r'\b\w+\b', claim.lower()))
    evid_tokens = set(re.findall(r'\b\w+\b', evidence.lower()))

    claim_has_neg = bool(claim_tokens.intersection(NEGATION_WORDS))
    evid_has_neg = bool(evid_tokens.intersection(NEGATION_WORDS))

    # Shared content words (excluding stop words)
    common_content = claim_tokens.intersection(evid_tokens) - NEGATION_WORDS
    if len(common_content) >= 3 and (claim_has_neg != evid_has_neg):
        return True
    return False


def detect_numeric_clash(claim: str, evidence: str) -> bool:
    """Detects if dates or numbers mentioned in claim clash with evidence."""
    claim_years = set(YEAR_PATTERN.findall(claim))
    evid_years = set(YEAR_PATTERN.findall(evidence))
    if claim_years and evid_years and not claim_years.intersection(evid_years):
        return True

    claim_nums = set(NUMBER_PATTERN.findall(claim)) - claim_years
    evid_nums = set(NUMBER_PATTERN.findall(evidence)) - evid_years
    if claim_nums and evid_nums and not claim_nums.intersection(evid_nums):
        return True

    return False


STOPWORDS = {"the", "a", "an", "is", "was", "are", "were", "in", "on", "at", "to", "for", "of", "and", "or", "by", "with"}


class SemanticSignal(BaseSignal):
    """Signal S: Directional faithfulness with deterministic negation/number verification."""

    def __init__(
        self,
        verifier_model: str = "lytang/MiniCheck-Flan-T5-Large",
        version: str = "v1.0",
        cache_dir: str = "data/cache/signals"
    ):
        super().__init__(name="semantic", version=version, cache_dir=cache_dir)
        self.verifier_name = verifier_model
        self.verifier = None

    def _lazy_load_verifier(self):
        if self.verifier is None:
            try:
                from minicheck import MiniCheck
                self.verifier = MiniCheck(model_name=self.verifier_name, enable_prefix_caching=False)
            except Exception:
                self.verifier = False

    def _compute(
        self,
        claim_text: str,
        evidence_text: str = "",
        query_text: str = "",
        samples: Optional[List[str]] = None
    ) -> SignalResult:
        t0 = time.time()
        claim_clean = claim_text.strip()
        evidence_clean = evidence_text.strip()

        if not claim_clean or not evidence_clean:
            return SignalResult(score=np.nan, available=False, latency_ms=0.0)

        # 1. Deterministic symbolic checks
        neg_clash = detect_negation_clash(claim_clean, evidence_clean)
        num_clash = detect_numeric_clash(claim_clean, evidence_clean)

        # 2. Neural directional faithfulness
        self._lazy_load_verifier()
        if self.verifier:
            try:
                # MiniCheck returns pred_labels and raw_probabilities
                _, raw_probs = self.verifier.score(docs=[evidence_clean], claims=[claim_clean])
                # Probability of being supported by doc
                p_supported = raw_probs[0]
                base_risk = 1.0 - p_supported
            except Exception:
                base_risk = 0.5
        else:
            # Deterministic content-word & entity overlap fallback (excluding stopwords)
            c_words = set(re.findall(r'\b\w+\b', claim_clean.lower())) - STOPWORDS
            e_words = set(re.findall(r'\b\w+\b', evidence_clean.lower())) - STOPWORDS
            if not c_words:
                base_risk = 0.5
            else:
                overlap = len(c_words.intersection(e_words)) / len(c_words)
                base_risk = 1.0 - overlap

            # Entity-specific ungrounded penalty
            c_entities = set(re.findall(r'\b[A-Z][a-z]+\b', claim_clean))
            e_entities = set(re.findall(r'\b[A-Z][a-z]+\b', evidence_clean))
            if c_entities:
                missing_entities = c_entities - e_entities
                if missing_entities:
                    missing_ratio = len(missing_entities) / len(c_entities)
                    base_risk = max(base_risk, 0.4 + 0.5 * missing_ratio)

        # 3. Boost risk if deterministic contradiction detected
        if neg_clash:
            base_risk = max(base_risk, 0.85)
        if num_clash:
            base_risk = max(base_risk, 0.80)

        score = float(np.clip(base_risk, 0.0, 1.0))
        lat = (time.time() - t0) * 1000.0

        return SignalResult(
            score=score,
            available=True,
            latency_ms=lat,
            metadata={
                "negation_clash": neg_clash,
                "numeric_clash": num_clash
            }
        )
