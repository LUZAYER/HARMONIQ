"""
Group-Disjoint Stratified Partitioning for Hallucination Benchmarks.

Enforces:
- RAGTruth: Official train/test split. Dev carved as 15% of train groups stratified by task and label.
- HaluEval, TruthfulQA, WikiBio, Fava: 60% train / 15% dev / 25% test by group, stratified by label.
- Seed: 42 (strictly reproducible).
- Group-disjoint: All records sharing a group (source_id, question, prompt, etc.) are strictly in the same split.
"""

import os
import json
import random
from collections import defaultdict
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.data import DatasetRecord


def create_group_splits(
    records: List[DatasetRecord],
    seed: int = 42
) -> Dict[str, str]:
    """
    Assigns each record ID to 'train', 'dev', or 'test' using group-disjoint stratified splitting.
    Returns: Dict[record_id, split_name]
    """
    random.seed(seed)
    np.random.seed(seed)

    # 1. Separate records by dataset
    by_dataset: Dict[str, List[DatasetRecord]] = defaultdict(list)
    for r in records:
        by_dataset[r.dataset].append(r)

    record_to_split: Dict[str, str] = {}

    for ds_name, ds_records in by_dataset.items():
        if ds_name == "RAGTruth":
            # RAGTruth official split handling
            train_groups = defaultdict(list)
            test_groups = defaultdict(list)

            for r in ds_records:
                official = r.metadata.get("official_split", "train")
                if official == "test":
                    test_groups[r.group].append(r)
                else:
                    train_groups[r.group].append(r)

            # Assign test groups
            for g, recs in test_groups.items():
                for r in recs:
                    record_to_split[r.id] = "test"

            # Carve 15% dev from train groups, stratified by task and label
            # Group summary: (task, group_label)
            strata = defaultdict(list)
            for g, recs in train_groups.items():
                task = recs[0].task
                g_label = 1 if any(rec.label == 1 for rec in recs) else 0
                strata[(task, g_label)].append(g)

            for stratum_key, group_list in strata.items():
                random.shuffle(group_list)
                n_dev = max(1, int(round(len(group_list) * 0.15)))
                dev_set = set(group_list[:n_dev])
                for g in group_list:
                    split_assign = "dev" if g in dev_set else "train"
                    for r in train_groups[g]:
                        record_to_split[r.id] = split_assign

        else:
            # 60 / 15 / 25 group split stratified by group label
            group_to_recs = defaultdict(list)
            for r in ds_records:
                group_to_recs[r.group].append(r)

            strata = defaultdict(list)
            for g, recs in group_to_recs.items():
                g_label = 1 if any(rec.label == 1 for rec in recs) else 0
                strata[g_label].append(g)

            for g_label, group_list in strata.items():
                random.shuffle(group_list)
                n_total = len(group_list)
                n_dev = max(1, int(round(n_total * 0.15)))
                n_test = max(1, int(round(n_total * 0.25)))

                dev_groups = set(group_list[:n_dev])
                test_groups = set(group_list[n_dev:n_dev + n_test])

                for g in group_list:
                    if g in dev_groups:
                        split_assign = "dev"
                    elif g in test_groups:
                        split_assign = "test"
                    else:
                        split_assign = "train"

                    for r in group_to_recs[g]:
                        record_to_split[r.id] = split_assign

    return record_to_split


def compute_split_summary_table(
    records: List[DatasetRecord],
    split_map: Dict[str, str]
) -> pd.DataFrame:
    """
    Auto-generates summary table: N records, N groups, and prevalence (fraction positive)
    for every dataset and split.
    """
    rows = []
    # Collect statistics per (dataset, split)
    grouped = defaultdict(list)
    for r in records:
        sp = split_map.get(r.id, "unassigned")
        grouped[(r.dataset, sp)].append(r)

    # Sort datasets with RAGTruth first
    ds_order = ["RAGTruth", "HaluEval-QA", "TruthfulQA", "WikiBio-GPT3", "FavaMultiSamples"]
    splits = ["train", "dev", "test"]

    for ds in ds_order:
        for sp in splits:
            subset = grouped.get((ds, sp), [])
            if not subset:
                continue
            n_recs = len(subset)
            n_groups = len(set(r.group for r in subset))
            n_claims = sum(r.n_claims for r in subset)
            prev = np.mean([r.label for r in subset]) if n_recs > 0 else 0.0

            rows.append({
                "Dataset": ds,
                "Split": sp,
                "N_Responses": n_recs,
                "N_Groups": n_groups,
                "N_Claims": n_claims,
                "Prevalence": round(float(prev), 4)
            })

    # Add total pooled rows per split
    for sp in splits:
        sp_recs = [r for r in records if split_map.get(r.id) == sp]
        if sp_recs:
            rows.append({
                "Dataset": "Pooled Total",
                "Split": sp,
                "N_Responses": len(sp_recs),
                "N_Groups": len(set(r.group for r in sp_recs)),
                "N_Claims": sum(r.n_claims for r in sp_recs),
                "Prevalence": round(float(np.mean([r.label for r in sp_recs])), 4)
            })

    return pd.DataFrame(rows)
