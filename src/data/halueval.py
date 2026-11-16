"""
HaluEval QA Dataset Loader.

Source: HaluEval (EMNLP 2023).
Enforces:
- Grouping unit: question (paired factual and hallucinated answers must never straddle splits)
- Provided evidence: 'knowledge' field
- Ground truth binary labels: 0 = factual, 1 = hallucinated
"""

import os
import json
from typing import List, Optional
from src.data import DatasetRecord, ClaimItem
from src.data.ragtruth import split_sentences_clausal


def load_halueval_qa(
    data_path: str = "data/raw/halueval/qa_data.json",
    limit_questions: Optional[int] = None
) -> List[DatasetRecord]:
    """
    Loads paired HaluEval QA instances.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"HaluEval QA file not found at {data_path}")

    records = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for q_idx, line in enumerate(f):
            if limit_questions and q_idx >= limit_questions:
                break
            if not line.strip():
                continue

            item = json.loads(line)
            q = item.get("question", "").strip()
            k = item.get("knowledge", "").strip()
            right_ans = item.get("right_answer", "").strip()
            hallu_ans = item.get("hallucinated_answer", "").strip()
            group_id = f"halueval_q_{q_idx}"

            # 1. Factual instance
            if right_ans:
                claims = [
                    ClaimItem(
                        claim_id=f"halueval_{q_idx}_f_c{c_idx}",
                        text=c["text"],
                        label=0,
                        start_char=c["start_char"],
                        end_char=c["end_char"]
                    )
                    for c_idx, c in enumerate(split_sentences_clausal(right_ans))
                ]
                records.append(DatasetRecord(
                    id=f"halueval_{q_idx}_factual",
                    group=group_id,
                    dataset="HaluEval-QA",
                    task="QA",
                    generator="human/gpt-3.5",
                    query=q,
                    response=right_ans,
                    label=0,
                    grounding_evidence=k,
                    stochastic_samples=[],
                    claims=claims,
                    metadata={"pair_type": "factual", "q_idx": q_idx}
                ))

            # 2. Hallucinated instance
            if hallu_ans:
                claims = [
                    ClaimItem(
                        claim_id=f"halueval_{q_idx}_h_c{c_idx}",
                        text=c["text"],
                        label=1,
                        start_char=c["start_char"],
                        end_char=c["end_char"]
                    )
                    for c_idx, c in enumerate(split_sentences_clausal(hallu_ans))
                ]
                records.append(DatasetRecord(
                    id=f"halueval_{q_idx}_hallu",
                    group=group_id,
                    dataset="HaluEval-QA",
                    task="QA",
                    generator="chatgpt-synthetic",
                    query=q,
                    response=hallu_ans,
                    label=1,
                    grounding_evidence=k,
                    stochastic_samples=[],
                    claims=claims,
                    metadata={"pair_type": "hallucinated", "q_idx": q_idx}
                ))

    return records
