#!/usr/bin/env python3
"""
Generate Group-Disjoint Benchmark Splits and Stratification Summary.

Creates:
- data/splits.json: Mapping from record_id to split ('train', 'dev', 'test')
- results/split_summary.csv: Full table of responses, claims, groups, and prevalence
"""

import os
import sys
import json
import argparse
import pandas as pd
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import DatasetRecord
from src.data.ragtruth import load_ragtruth
from src.data.halueval import load_halueval_qa
from src.data.truthfulqa import load_truthfulqa
from src.data.wikibio import load_wikibio_gpt3
from src.data.fava import load_favamultisamples
from src.data.splits import create_group_splits, compute_split_summary_table


def load_all_benchmarks(
    halueval_limit: int = None,
    verbose: bool = True
) -> List[DatasetRecord]:
    """
    Loads all five benchmarks into standardized records.
    """
    all_records: List[DatasetRecord] = []

    # 1. RAGTruth
    if verbose:
        print("Loading RAGTruth...")
    ragtruth_records = load_ragtruth("data/raw/ragtruth", filter_good_quality=True)
    all_records.extend(ragtruth_records)
    if verbose:
        print(f"  RAGTruth: {len(ragtruth_records)} valid records loaded.")

    # 2. HaluEval QA
    if verbose:
        print("Loading HaluEval QA...")
    halueval_records = load_halueval_qa("data/raw/halueval/qa_data.json", limit_questions=halueval_limit)
    all_records.extend(halueval_records)
    if verbose:
        print(f"  HaluEval QA: {len(halueval_records)} paired records loaded.")

    # 3. TruthfulQA
    if verbose:
        print("Loading TruthfulQA...")
    truthfulqa_records = load_truthfulqa("data/raw/truthfulqa/TruthfulQA.csv")
    all_records.extend(truthfulqa_records)
    if verbose:
        print(f"  TruthfulQA: {len(truthfulqa_records)} paired records loaded.")

    # 4. WikiBio GPT-3
    if verbose:
        print("Loading WikiBio GPT-3...")
    wikibio_records = load_wikibio_gpt3("data/raw/selfcheckgpt/wiki_bio_gpt3_hallucination.json")
    all_records.extend(wikibio_records)
    if verbose:
        print(f"  WikiBio GPT-3: {len(wikibio_records)} records loaded.")

    # 5. FAVA MultiSamples
    if verbose:
        print("Loading FAVA MultiSamples...")
    fava_records = load_favamultisamples("data/raw/favamultisamples/fava_multisamples.parquet")
    all_records.extend(fava_records)
    if verbose:
        print(f"  FAVA MultiSamples: {len(fava_records)} records loaded.")

    return all_records


def main():
    parser = argparse.ArgumentParser(description="Create group-disjoint splits across benchmarks.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for splitting")
    parser.add_argument('--output-json', type=str, default="data/splits.json", help="Path to splits.json")
    parser.add_argument('--summary-csv', type=str, default="results/split_summary.csv", help="Summary CSV")
    parser.add_argument('--halueval-sample', type=int, default=None,
                        help="Optional limit for fast local testing")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True)

    records = load_all_benchmarks(halueval_limit=args.halueval_sample, verbose=True)
    print(f"\nTotal loaded records: {len(records)}")

    # Generate splits
    split_map = create_group_splits(records, seed=args.seed)

    # Verification: Ensure strict group disjointness
    group_splits = {}
    for r in records:
        sp = split_map[r.id]
        if r.group in group_splits:
            assert group_splits[r.group] == sp, f"Leakage detected! Group {r.group} split across {group_splits[r.group]} and {sp}"
        else:
            group_splits[r.group] = sp
    print("✓ Strict group-disjointness verified: 0 group leakage across splits.")

    # Save data/splits.json
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(split_map, f, indent=2)
    print(f"✓ Saved {args.output_json} with {len(split_map)} assignments.")

    # Compute summary table
    summary_df = compute_split_summary_table(records, split_map)
    summary_df.to_csv(args.summary_csv, index=False)
    print(f"✓ Saved split summary to {args.summary_csv}")
    print("\n" + summary_df.to_string(index=False))


if __name__ == '__main__':
    main()
