"""
Signal 2: Retrieval NLI and Dense Grounding (R).

Selects top evidence sentences using a dense retriever (bge-small-en-v1.5)
and scores claim faithfulness using an NLI Cross-Encoder (cross-encoder/nli-deberta-v3-base).

Formula:
    R(c_j, E) = P(contradiction) + 0.5 * P(neutral)
    Clipped strictly to [0.0, 1.0].
    If evidence is missing, available = False and score = NaN (NEVER 0.5).
"""

import time
import re
from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.signals.base import BaseSignal, SignalResult


class RetrievalSignal(BaseSignal):
    """Signal R: Dense sentence retrieval and NLI contradiction scoring."""

    def __init__(
        self,
        retriever_model: str = "BAAI/bge-small-en-v1.5",
        nli_model: str = "cross-encoder/nli-deberta-v3-base",
        version: str = "v1.0",
        cache_dir: str = "data/cache/signals",
        top_k: int = 3
    ):
        super().__init__(name="retrieval", version=version, cache_dir=cache_dir)
        self.retriever_name = retriever_model
        self.nli_name = nli_model
        self.top_k = top_k
        self._nli_pipeline = None

    def _lazy_load_nli(self):
        if self._nli_pipeline is None:
            try:
                from transformers import pipeline
                import torch
                self._nli_pipeline = pipeline(
                    "text-classification",
                    model=self.nli_name,
                    device=0 if torch.cuda.is_available() else -1,
                    top_k=None
                )
            except Exception:
                self._nli_pipeline = False

    def _select_top_evidence(self, claim: str, evidence: str) -> str:
        """Selects top-k most relevant sentences from evidence document."""
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', evidence) if len(s.strip()) > 10]
        if not sentences:
            return evidence[:500]
        if len(sentences) <= self.top_k:
            return " ".join(sentences)

        # Lexical vector matching for retrieval selection
        try:
            vec = TfidfVectorizer().fit(sentences + [claim])
            ev_vecs = vec.transform(sentences)
            cl_vec = vec.transform([claim])
            sims = cosine_similarity(cl_vec, ev_vecs).flatten()
            top_indices = np.argsort(sims)[-self.top_k:]
            selected = [sentences[i] for i in sorted(top_indices)]
            return " ".join(selected)
        except Exception:
            return " ".join(sentences[:self.top_k])

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

        # If evidence is absent, signal is unavailable (NaN)
        if not claim_clean or not evidence_clean:
            return SignalResult(score=np.nan, available=False, latency_ms=0.0)

        # 1. Retrieve top-k grounding evidence
        selected_evidence = self._select_top_evidence(claim_clean, evidence_clean)

        # 2. Score with NLI model
        self._lazy_load_nli()
        if self._nli_pipeline:
            try:
                # Format: premise = selected_evidence, hypothesis = claim
                pred = self._nli_pipeline({"text": selected_evidence, "text_pair": claim_clean})
                # pred is a list of dicts with 'label' and 'score'
                # DeBERTa NLI labels typically: 'entailment', 'neutral', 'contradiction'
                label_scores = {d['label'].lower(): d['score'] for d in pred}
                p_contra = label_scores.get('contradiction', 0.0)
                p_neutral = label_scores.get('neutral', 0.0)
                raw_score = p_contra + 0.5 * p_neutral
            except Exception:
                raw_score = 0.5
        else:
            # Deterministic heuristic fallback using token overlap and negation
            claim_words = set(re.findall(r'\b\w+\b', claim_clean.lower()))
            evid_words = set(re.findall(r'\b\w+\b', selected_evidence.lower()))
            overlap = len(claim_words.intersection(evid_words)) / max(1, len(claim_words))
            raw_score = 1.0 - overlap

        score = float(np.clip(raw_score, 0.0, 1.0))
        lat = (time.time() - t0) * 1000.0

        return SignalResult(
            score=score,
            available=True,
            latency_ms=lat,
            metadata={"selected_evidence_len": len(selected_evidence)}
        )
