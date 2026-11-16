"""
WikiBio GPT-3 Dataset Loader (SelfCheckGPT Benchmark).

Source: Manakul et al. (EMNLP 2023).
Contains 238 GPT-3 generated biographies with sentence-level human annotations
and stochastic generation samples for consistency analysis.
Enforces:
- Grouping unit: wiki_bio_test_idx
- Human sentence annotations: accurate (0) vs minor/major_inaccurate (1)
- Stochastic samples preserved for Self-Verification (V)
"""

import os
import json
from typing import List, Optional
from src.data import DatasetRecord, ClaimItem


def load_wikibio_gpt3(
    data_path: str = "data/raw/selfcheckgpt/wiki_bio_gpt3_hallucination.json",
    limit: Optional[int] = None
) -> List[DatasetRecord]:
    """
    Loads WikiBio GPT-3 benchmark instances.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"WikiBio file not found at {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = []
    for idx, item in enumerate(data):
        if limit and idx >= limit:
            break

        test_idx = item.get("wiki_bio_test_idx", idx)
        gpt3_text = item.get("gpt3_text", "").strip()
        wiki_bio_text = item.get("wiki_bio_text", "").strip()
        sentences = item.get("gpt3_sentences", [])
        annotations = item.get("annotation", [])
        text_samples = item.get("gpt3_text_samples", [])

        # Label definitions: inaccurate (minor or major) -> 1, accurate -> 0
        has_hallu = 1 if any("inaccurate" in str(a).lower() for a in annotations) else 0

        claims = []
        for s_idx, (s, a) in enumerate(zip(sentences, annotations)):
            is_inacc = 1 if "inaccurate" in str(a).lower() else 0
            claims.append(ClaimItem(
                claim_id=f"wikibio_{test_idx}_c{s_idx}",
                text=str(s).strip(),
                label=is_inacc,
                metadata={"raw_annotation": str(a)}
            ))

        records.append(DatasetRecord(
            id=f"wikibio_{test_idx}",
            group=f"wikibio_idx_{test_idx}",
            dataset="WikiBio-GPT3",
            task="Bio",
            generator="gpt-3-text-davinci-003",
            query=f"Generate biography for Wikipedia concept #{test_idx}",
            response=gpt3_text,
            label=has_hallu,
            grounding_evidence=wiki_bio_text,
            stochastic_samples=text_samples,
            claims=claims,
            metadata={"test_idx": test_idx}
        ))

    return records
