"""
Unit tests for HARMONIQ signal stack on 10 hand-crafted evaluation claims.

Verifies:
- Confidence (C), Retrieval (R), Semantic (S), Self-Verification (V) behavior.
- Missing signals strictly return NaN and available=False (NEVER 0.5).
- Caching to Parquet preserves exact scores.
- Deterministic contradiction detection on polarity flips and date/number clashes.
"""

import os
import unittest
import numpy as np

from src.signals.confidence import ConfidenceSignal
from src.signals.retrieval import RetrievalSignal
from src.signals.semantic import SemanticSignal
from src.signals.selfverif import SelfVerificationSignal
from src.signals.base import compute_claim_hash


class TestSignals(unittest.TestCase):
    """Test suite for claim-level signals on known ground truth."""

    @classmethod
    def setUpClass(cls):
        cls.cache_dir = "data/cache/test_signals"
        os.makedirs(cls.cache_dir, exist_ok=True)
        cls.sig_c = ConfidenceSignal(cache_dir=cls.cache_dir)
        cls.sig_r = RetrievalSignal(cache_dir=cls.cache_dir)
        cls.sig_s = SemanticSignal(cache_dir=cls.cache_dir)
        cls.sig_v = SelfVerificationSignal(cache_dir=cls.cache_dir)

    def test_claim_1_factual_matching_evidence(self):
        """Claim 1: Supported factual claim must have low hallucination risk in R and S."""
        evidence = "The Eiffel Tower is an iron lattice tower located on the Champ de Mars in Paris, France."
        claim = "The Eiffel Tower is located in Paris, France."
        res_r = self.sig_r.evaluate_claim(claim, evidence)
        res_s = self.sig_s.evaluate_claim(claim, evidence)

        self.assertTrue(res_r.available)
        self.assertTrue(res_s.available)
        self.assertLess(res_r.score, 0.40, "Factual claim should have low retrieval hallucination score")
        self.assertLess(res_s.score, 0.40, "Factual claim should have low semantic hallucination score")

    def test_claim_2_contradiction_evidence(self):
        """Claim 2: Contradicted claim must have high hallucination risk."""
        evidence = "The Eiffel Tower is an iron lattice tower located in Paris, France."
        claim = "The Eiffel Tower is located in Berlin, Germany."
        res_s = self.sig_s.evaluate_claim(claim, evidence)

        self.assertTrue(res_s.available)
        self.assertGreater(res_s.score, 0.50, "Contradicted claim should have elevated risk score")

    def test_claim_3_numerical_date_clash(self):
        """Claim 3: Numerical date mismatch must trigger elevated risk in S."""
        evidence = "The Apollo 11 mission landed on the Moon in July 1969."
        claim = "The Apollo 11 mission landed on the Moon in 1985."
        res_s = self.sig_s.evaluate_claim(claim, evidence)

        self.assertTrue(res_s.available)
        self.assertTrue(res_s.metadata.get("numeric_clash", False), "Must detect date mismatch")
        self.assertGreaterEqual(res_s.score, 0.75, "Date mismatch must yield high risk score")

    def test_claim_4_negation_polarity_clash(self):
        """Claim 4: Negation polarity flip must trigger elevated risk in S."""
        evidence = "The clinical trial confirmed that the drug was effective and safe."
        claim = "The clinical trial showed that the drug was not effective."
        res_s = self.sig_s.evaluate_claim(claim, evidence)

        self.assertTrue(res_s.available)
        self.assertTrue(res_s.metadata.get("negation_clash", False), "Must detect negation polarity clash")
        self.assertGreaterEqual(res_s.score, 0.75, "Negation mismatch must yield high risk score")

    def test_claim_5_epistemic_hedging(self):
        """Claim 5: Heavy hedging markers must elevate confidence uncertainty in C."""
        claim_confident = "Albert Einstein published the general theory of relativity in 1915."
        claim_hedged = "Perhaps Albert Einstein might have allegedly published the theory of relativity."

        res_conf = self.sig_c.evaluate_claim(claim_confident)
        res_hedged = self.sig_c.evaluate_claim(claim_hedged)

        self.assertTrue(res_conf.available)
        self.assertTrue(res_hedged.available)
        self.assertGreater(res_hedged.score, res_conf.score, "Hedged claim must have higher uncertainty score")

    def test_claim_6_missing_evidence_returns_nan(self):
        """Claim 6: When evidence is missing, R and S must return NaN and available=False (never 0.5)."""
        claim = "Photosynthesis converts carbon dioxide into glucose."
        res_r = self.sig_r.evaluate_claim(claim, evidence_text="")
        res_s = self.sig_s.evaluate_claim(claim, evidence_text="")

        self.assertFalse(res_r.available)
        self.assertTrue(np.isnan(res_r.score), "Unavailable R must be NaN, not 0.5")
        self.assertFalse(res_s.available)
        self.assertTrue(np.isnan(res_s.score), "Unavailable S must be NaN, not 0.5")

    def test_claim_7_missing_samples_returns_nan(self):
        """Claim 7: When stochastic samples are absent, V must return NaN and available=False."""
        claim = "Water boils at 100 degrees Celsius at standard atmospheric pressure."
        res_v = self.sig_v.evaluate_claim(claim, samples=[])

        self.assertFalse(res_v.available)
        self.assertTrue(np.isnan(res_v.score), "Unavailable V must be NaN, not 0.5")

    def test_claim_8_agreeing_stochastic_samples(self):
        """Claim 8: Consistent stochastic samples produce low hallucination score in V."""
        claim = "Alan Turing developed the concept of the universal machine."
        samples = [
            "Alan Turing introduced the theoretical concept of the universal Turing machine.",
            "In 1936, Turing proved that a universal machine could compute any computable sequence.",
            "The universal machine concept was formulated by British mathematician Alan Turing."
        ]
        res_v = self.sig_v.evaluate_claim(claim, samples=samples)

        self.assertTrue(res_v.available)
        self.assertLess(res_v.score, 0.40, "Agreeing samples should produce low risk in V")

    def test_claim_9_disagreeing_stochastic_samples(self):
        """Claim 9: Conflicting/divergent stochastic samples produce high hallucination score in V."""
        claim = "The author was born in Tokyo in 1950."
        samples = [
            "The author was born in Osaka and raised in Kyoto.",
            "Records indicate the author was born in Hiroshima in 1962.",
            "She spent her early childhood in Sapporo before moving abroad."
        ]
        res_v = self.sig_v.evaluate_claim(claim, samples=samples)

        self.assertTrue(res_v.available)
        self.assertGreater(res_v.score, 0.60, "Disagreeing samples must produce elevated risk in V")

    def test_claim_10_caching_consistency(self):
        """Claim 10: Repeated evaluation of same claim retrieves exact score from cache."""
        claim = "Python was created by Guido van Rossum."
        res1 = self.sig_c.evaluate_claim(claim)
        res2 = self.sig_c.evaluate_claim(claim)

        self.assertEqual(res1.score, res2.score)
        self.assertTrue(res2.metadata.get("from_cache", False))


if __name__ == '__main__':
    unittest.main()
