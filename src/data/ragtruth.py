"""
RAGTruth Dataset Loader.

Source: ParticleMedia/RAGTruth (ACL 2024).
Contains 17,790 responses over 2,965 sources across 6 generators and 3 tasks (QA, Summary, Data2txt).
Enforces:
- Drop non-good quality responses (incorrect_refusal, truncated)
- Grouping unit: source_id (strictly disjoint)
- Span-level character overlap label propagation
- Ground truth binary labels: 0 = factual, 1 = hallucinated
"""

import os
import json
import re
from typing import List, Dict, Any, Optional
from src.data import DatasetRecord, ClaimItem


def split_sentences_clausal(text: str) -> List[Dict[str, Any]]:
    """
    Decompose text into clausal/sentence-level units with character span offsets.
    """
    spans = []
    # Match sentences ending with punctuation, or newlines
    pattern = re.compile(r'[^.!?\n]+[.!?\n]?')
    for match in pattern.finditer(text):
        claim_str = match.group().strip()
        if len(claim_str) > 3:  # avoid bare punctuation
            spans.append({
                "text": claim_str,
                "start_char": match.start(),
                "end_char": match.end()
            })
    if not spans and text.strip():
        spans.append({
            "text": text.strip(),
            "start_char": 0,
            "end_char": len(text)
        })
    return spans


def check_span_overlap(claim_start: int, claim_end: int, hallu_spans: List[Dict[str, Any]]) -> bool:
    """
    Returns True if the claim span overlaps any gold hallucination span.
    """
    for span in hallu_spans:
        h_start = span.get("start", 0)
        h_end = span.get("end", 0)
        # Check interval intersection
        if max(claim_start, h_start) < min(claim_end, h_end):
            return True
    return False


def load_ragtruth(
    data_dir: str = "data/raw/ragtruth",
    filter_good_quality: bool = True
) -> List[DatasetRecord]:
    """
    Loads RAGTruth dataset into standardized DatasetRecord instances.
    """
    source_path = os.path.join(data_dir, "source_info.jsonl")
    response_path = os.path.join(data_dir, "response.jsonl")

    if not os.path.exists(source_path) or not os.path.exists(response_path):
        raise FileNotFoundError(f"RAGTruth files not found in {data_dir}")

    # 1. Load sources
    sources: Dict[int, Dict[str, Any]] = {}
    with open(source_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            sources[item["source_id"]] = item

    # 2. Load responses
    records = []
    dropped_counts = {"incorrect_refusal": 0, "truncated": 0, "other": 0}

    with open(response_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)

            quality = item.get("quality", "good")
            if filter_good_quality and quality != "good":
                if quality in dropped_counts:
                    dropped_counts[quality] += 1
                else:
                    dropped_counts["other"] += 1
                continue

            resp_id = str(item.get("id"))
            source_id = str(item.get("source_id"))
            source_meta = sources.get(int(source_id), {})

            task_type = source_meta.get("task_type", "Summary")
            prompt = source_meta.get("prompt", "")
            evidence = source_meta.get("source", "")
            response_text = item.get("response", "")
            generator = item.get("model", "unknown")
            gold_spans = item.get("labels", [])

            # Response-level label: positive if any hallucination span exists
            resp_label = 1 if len(gold_spans) > 0 else 0

            # Decompose into atomic claims with span offsets
            raw_claims = split_sentences_clausal(response_text)
            claim_items = []
            for c_idx, c in enumerate(raw_claims):
                is_hallu = 1 if check_span_overlap(c["start_char"], c["end_char"], gold_spans) else 0
                claim_items.append(ClaimItem(
                    claim_id=f"ragtruth_{resp_id}_c{c_idx}",
                    text=c["text"],
                    label=is_hallu,
                    start_char=c["start_char"],
                    end_char=c["end_char"]
                ))

            rec = DatasetRecord(
                id=f"ragtruth_{resp_id}",
                group=f"ragtruth_source_{source_id}",
                dataset="RAGTruth",
                task=task_type,
                generator=generator,
                query=prompt,
                response=response_text,
                label=resp_label,
                grounding_evidence=evidence,
                stochastic_samples=[],  # RAGTruth does not provide multi-samples
                claims=claim_items,
                metadata={
                    "official_split": item.get("split"),
                    "quality": quality,
                    "temperature": item.get("temperature", 0.0),
                    "source_id": source_id,
                    "gold_spans": gold_spans
                }
            )
            records.append(rec)

    return records
