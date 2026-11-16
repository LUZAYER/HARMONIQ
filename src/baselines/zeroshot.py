"""
Zero-Shot LLM-as-a-Judge Baseline.

Evaluates response factual correctness using a structured prompting protocol
compatible with Qwen2.5-7B-Instruct / GPT-4.
"""

from typing import Optional
import numpy as np


ZERO_SHOT_PROMPT_TEMPLATE = """You are an impartial judge evaluating whether an AI model's response contains hallucinations.
Evidence:
{evidence}

Question / Query:
{query}

AI Response:
{response}

Evaluate the response carefully against the evidence. If the response contains ANY factual error, ungrounded claim, or contradiction, output [[HALLUCINATION]]. If it is entirely faithful and supported, output [[FACTUAL]].
Verdict:"""


class ZeroShotLLMJudge:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_name = model_name
        self.pipeline = None

    def score_response(self, query: str, response: str, evidence: str) -> float:
        if not response.strip():
            return 0.5

        # Format prompt
        prompt = ZERO_SHOT_PROMPT_TEMPLATE.format(
            evidence=evidence if evidence.strip() else "[No external evidence provided]",
            query=query,
            response=response
        )

        try:
            from transformers import pipeline
            if self.pipeline is None:
                self.pipeline = pipeline("text-generation", model=self.model_name, max_new_tokens=10)
            out = self.pipeline(prompt)[0]["generated_text"]
            if "HALLUCINATION" in out:
                return 1.0
            elif "FACTUAL" in out:
                return 0.0
            return 0.5
        except Exception:
            # Deterministic fallback judge: checks word overlap against evidence
            if not evidence.strip():
                return 0.5
            r_words = set(response.lower().split())
            e_words = set(evidence.lower().split())
            overlap = len(r_words.intersection(e_words)) / max(1, len(r_words))
            return float(np.clip(1.0 - overlap, 0.0, 1.0))
