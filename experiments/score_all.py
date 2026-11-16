#!/usr/bin/env python3
"""
Score Claims Across Benchmarks Using HARMONIQ Signal Stack.

Generates:
- results/claims.parquet: Claim-level dataset with extracted signals and metadata.

Supports:
- --smoke-test (200 records) for rapid validation under 1 minute.
- Persistent parquet signal caching in data/cache/signals/.
"""

import os
import sys
import argparse
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.ragtruth import load_ragtruth
from src.data.halueval import load_halueval_qa
from src.data.truthfulqa import load_truthfulqa
from src.data.wikibio import load_wikibio_gpt3
from src.data.fava import load_favamultisamples
from src.signals.confidence import ConfidenceSignal, extract_critical_tokens
from src.signals.retrieval import RetrievalSignal
from src.signals.semantic import SemanticSignal
from src.signals.selfverif import SelfVerificationSignal


def run_claim_scoring(
    records: list,
    split_map: dict,
    output_parquet: str = "results/claims.parquet",
    batch_size: int = 100
) -> pd.DataFrame:
    """
    Extracts signals across all claims in the provided records.
    """
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)

    sig_c = ConfidenceSignal()
    sig_r = RetrievalSignal()
    sig_s = SemanticSignal()
    sig_v = SelfVerificationSignal()

    rows = []
    print(f"Scoring claims for {len(records)} records...")
    t0 = time.time()

    for r in tqdm(records, desc="Extracting signals"):
        sp = split_map.get(r.id, "train")

        for claim in r.claims:
            c_text = claim.text
            evid = r.grounding_evidence
            samples = r.stochastic_samples

            # 1. Signal C (Confidence)
            res_c = sig_c.evaluate_claim(c_text, evid, r.query, samples)
            # 2. Signal R (Retrieval NLI)
            res_r = sig_r.evaluate_claim(c_text, evid, r.query, samples)
            # 3. Signal S (Semantic Alignment)
            res_s = sig_s.evaluate_claim(c_text, evid, r.query, samples)
            # 4. Signal V (Self-Verification)
            res_v = sig_v.evaluate_claim(c_text, evid, r.query, samples)

            critical_ents = extract_critical_tokens(c_text)

            rows.append({
                "response_id": r.id,
                "claim_id": claim.claim_id,
                "group": r.group,
                "dataset": r.dataset,
                "task": r.task,
                "generator": r.generator,
                "split": sp,
                "claim_text": c_text,
                "claim_label": claim.label,
                "response_label": r.label,
                "sig_C": res_c.score,
                "avail_C": int(res_c.available),
                "sig_R": res_r.score,
                "avail_R": int(res_r.available),
                "sig_S": res_s.score,
                "avail_S": int(res_s.available),
                "sig_V": res_v.score,
                "avail_V": int(res_v.available),
                "claim_len": len(c_text.split()),
                "entity_count": len(critical_ents)
            })

    # Flush caches to disk
    sig_c.cache.flush()
    sig_r.cache.flush()
    sig_s.cache.flush()
    sig_v.cache.flush()

    df = pd.DataFrame(rows)
    df.to_parquet(output_parquet, index=False)
    elapsed = time.time() - t0
    print(f"✓ Scored {len(df)} claims across {len(records)} records in {elapsed:.2f}s")
    print(f"✓ Saved to {output_parquet}")
    return df


def main():
    import json
    parser = argparse.ArgumentParser(description="Extract claim-level signals across benchmarks.")
    parser.add_argument('--smoke-test', action='store_true', help="Run 200 records quick smoke test")
    parser.add_argument('--limit', type=int, default=None, help="Record limit")
    parser.add_argument('--splits', type=str, default="data/splits.json", help="Path to splits.json")
    parser.add_argument('--output', type=str, default="results/claims.parquet", help="Output claims parquet")
    args = parser.parse_args()

    with open(args.splits, 'r', encoding='utf-8') as f:
        split_map = json.load(f)

    if args.smoke_test:
        print("Running 200-record smoke test...")
        # 50 from each dataset
        records = (
            load_ragtruth()[:50] +
            load_halueval_qa(limit_questions=25) +
            load_truthfulqa(limit_questions=25) +
            load_wikibio_gpt3()[:50] +
            load_favamultisamples()[:50]
        )
    else:
        # Full benchmark loading
        from experiments.make_splits import load_all_benchmarks
        records = load_all_benchmarks(halueval_limit=args.limit)

    run_claim_scoring(records, split_map, output_parquet=args.output)


if __name__ == '__main__':
    main()
