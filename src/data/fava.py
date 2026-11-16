"""
FAVA MultiSamples Dataset Loader (FactSelfCheck 2026 / EACL).

Contains 460 generations across 150 prompts with sentence-level binary annotations
and stochastic candidate generations.
Enforces:
- Grouping unit: prompt (strictly disjoint across splits)
- Sentence binary annotations (0 = factual, 1 = hallucinated)
- Preservation of stochastic samples for self-verification
"""

import os
import pandas as pd
from typing import List, Optional
from src.data import DatasetRecord, ClaimItem


def load_favamultisamples(
    data_path: str = "data/raw/favamultisamples/fava_multisamples.parquet",
    limit: Optional[int] = None
) -> List[DatasetRecord]:
    """
    Loads FAVA multi-sample instances with sentence-level annotations.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"FAVA parquet file not found at {data_path}")

    df = pd.read_parquet(data_path)
    records = []

    for idx, row in df.iterrows():
        if limit and len(records) >= limit:
            break

        prompt = str(row.get("prompt", "")).strip()
        output = str(row.get("output", "")).strip()
        sentences = row.get("sentences", [])
        bin_annots = row.get("sentences_binary_annotations", [])
        text_samples = row.get("text_samples", [])
        if hasattr(text_samples, "tolist"):
            text_samples = text_samples.tolist()
        elif not isinstance(text_samples, list):
            text_samples = []

        has_hallu = 1 if any(b == 1 for b in bin_annots) else 0

        claims = []
        if len(sentences) == len(bin_annots):
            for s_idx, (s, b) in enumerate(zip(sentences, bin_annots)):
                claims.append(ClaimItem(
                    claim_id=f"fava_{idx}_c{s_idx}",
                    text=str(s).strip(),
                    label=int(b)
                ))

        records.append(DatasetRecord(
            id=f"fava_{idx}",
            group=f"fava_prompt_{hash(prompt) % 1000000}",
            dataset="FavaMultiSamples",
            task="General",
            generator="llama/chatgpt",
            query=prompt,
            response=output,
            label=has_hallu,
            grounding_evidence="",
            stochastic_samples=text_samples,
            claims=claims,
            metadata={"origin": row.get("dataset", "FAVA"), "row_idx": idx}
        ))

    return records
