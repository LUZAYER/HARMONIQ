"""
Unit tests for benchmark data loaders and group-disjoint splits.

Verifies:
- Official partition counts and quality filtering match Section 3 ground truth.
- TruthfulQA has zero answer-key leakage in grounding evidence.
- Group-disjointness holds strictly across train, dev, and test.
"""

import os
import unittest
import pandas as pd
from collections import defaultdict

from src.data.ragtruth import load_ragtruth
from src.data.halueval import load_halueval_qa
from src.data.truthfulqa import load_truthfulqa
from src.data.wikibio import load_wikibio_gpt3
from src.data.fava import load_favamultisamples
from src.data.splits import create_group_splits


class TestDataLoaders(unittest.TestCase):
    """Test suite for benchmark data loading and partitioning."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_ragtruth_counts_and_quality(self):
        """Verify RAGTruth matches Section 3: 17,617 good responses, 2,965 sources, official train/test."""
        records = load_ragtruth(filter_good_quality=True)
        self.assertEqual(len(records), 17617, "RAGTruth good responses count must equal 17,617")

        train_recs = [r for r in records if r.metadata.get("official_split") == "train"]
        test_recs = [r for r in records if r.metadata.get("official_split") == "test"]
        self.assertEqual(len(train_recs), 14942, "RAGTruth train count must equal 14,942")
        self.assertEqual(len(test_recs), 2675, "RAGTruth test count must equal 2,675")

        unique_sources = set(r.metadata.get("source_id") for r in records)
        self.assertEqual(len(unique_sources), 2965, "RAGTruth must contain exactly 2,965 unique sources")

        test_prev = sum(r.label for r in test_recs) / len(test_recs)
        self.assertAlmostEqual(test_prev, 0.353, places=2, msg="Test prevalence must be ~0.353")

    def test_truthfulqa_zero_leakage(self):
        """Verify TruthfulQA has 790 questions and never uses Correct Answers as grounding evidence."""
        records = load_truthfulqa()
        # 790 questions * 2 responses (best and incorrect) = 1580
        self.assertEqual(len(records), 1580)
        unique_groups = set(r.group for r in records)
        self.assertEqual(len(unique_groups), 790)

        # Check evidence: must NOT leak answer key
        for r in records:
            self.assertEqual(
                r.grounding_evidence, "",
                "TruthfulQA grounding evidence must be empty (regime b only, zero answer key leakage)"
            )

    def test_wikibio_counts_and_samples(self):
        """Verify WikiBio contains 238 items with stochastic samples preserved."""
        records = load_wikibio_gpt3()
        self.assertEqual(len(records), 238)
        for r in records:
            self.assertTrue(len(r.stochastic_samples) > 0, "WikiBio must have stochastic samples for V signal")
            self.assertTrue(len(r.claims) > 0, "WikiBio must have decomposed sentence claims")

    def test_fava_counts(self):
        """Verify FAVA MultiSamples contains 460 items across 150 prompt groups."""
        records = load_favamultisamples()
        self.assertEqual(len(records), 460)
        unique_prompts = set(r.query for r in records)
        self.assertEqual(len(unique_prompts), 150)

    def test_group_disjoint_splitting(self):
        """Verify that splitting never places records from the same group across different splits."""
        # Test on a mix of records
        sample_recs = (
            load_wikibio_gpt3()[:50] +
            load_favamultisamples()[:50] +
            load_truthfulqa()[:50]
        )
        split_map = create_group_splits(sample_recs, seed=42)

        group_splits = defaultdict(set)
        for r in sample_recs:
            sp = split_map[r.id]
            group_splits[r.group].add(sp)

        # Assert no group has >1 split
        for group, splits in group_splits.items():
            self.assertEqual(
                len(splits), 1,
                f"Data leakage detected! Group {group} present in multiple splits: {splits}"
            )


if __name__ == '__main__':
    unittest.main()
