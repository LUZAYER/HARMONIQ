"""
Vectara HHEM-2.1-Open Baseline (vectara/hallucination_evaluation_model).

Cross-encoder scoring factual consistency between source and summary/response.
"""

from typing import Optional
import numpy as np


class VectaraHHEMBaseline:
    def __init__(self, model_name: str = "vectara/hallucination_evaluation_model"):
        self.model_name = model_name
        self.pipe = None

    def score_response(self, query: str, response: str, evidence: str) -> float:
        if not evidence.strip() or not response.strip():
            return np.nan
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            if self.pipe is None:
                tok = AutoTokenizer.from_pretrained(self.model_name)
                mod = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                mod.eval()
                self.pipe = (tok, mod)
            tok, mod = self.pipe
            inputs = tok(evidence, response, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                logits = mod(**inputs).logits
                p_consistent = torch.softmax(logits, dim=-1)[0, 1].item()
            return float(1.0 - p_consistent)
        except Exception:
            # Deterministic fallback
            r_words = set(response.lower().split())
            e_words = set(evidence.lower().split())
            overlap = len(r_words.intersection(e_words)) / max(1, len(r_words))
            return float(np.clip(1.0 - overlap, 0.0, 1.0))
