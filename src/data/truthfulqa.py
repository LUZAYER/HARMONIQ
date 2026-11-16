"""
TruthfulQA Dataset Loader.

Source: Lin et al. (ACL 2022).
Enforces:
- Strict zero-leakage: NEVER use 'Correct Answers' as grounding evidence.
- Evaluated exclusively under retrieved evidence regime (b).
- Grouping unit: Question (factual Best Answer and hallucinated Incorrect Answer kept in same split).
"""

import os
import pandas as pd
from typing import List, Optional
from src.data import DatasetRecord, ClaimItem
from src.data.ragtruth import split_sentences_clausal


def load_truthfulqa(
    data_path: str = "data/raw/truthfulqa/TruthfulQA.csv",
    limit_questions: Optional[int] = None
) -> List[DatasetRecord]:
    """
    Loads TruthfulQA pairs with zero answer-key leakage.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"TruthfulQA CSV not found at {data_path}")

    df = pd.read_csv(data_path)
    records = []

    for idx, row in df.iterrows():
        if limit_questions and idx >= limit_questions:
            break

        q = str(row.get("Question", "")).strip()
        best_ans = str(row.get("Best Answer", "")).strip()
        
        # Best incorrect answer or first incorrect
        incorrect_col = row.get("Best Incorrect Answer")
        if pd.isna(incorrect_col) or not str(incorrect_col).strip():
            inc_list = str(row.get("Incorrect Answers", "")).split(";")
            incorrect_ans = inc_list[0].strip() if inc_list else ""
        else:
            incorrect_ans = str(incorrect_col).strip()

        group_id = f"truthfulqa_q_{idx}"
        category = str(row.get("Category", "General"))

        # Zero leakage: grounding_evidence is explicitly empty (only retrieved evidence allowed in regime b)
        grounding_evidence = ""

        # 1. Factual instance
        if best_ans and best_ans != "nan":
            claims = [
                ClaimItem(
                    claim_id=f"truthfulqa_{idx}_f_c{c_idx}",
                    text=c["text"],
                    label=0,
                    start_char=c["start_char"],
                    end_char=c["end_char"]
                )
                for c_idx, c in enumerate(split_sentences_clausal(best_ans))
            ]
            records.append(DatasetRecord(
                id=f"truthfulqa_{idx}_factual",
                group=group_id,
                dataset="TruthfulQA",
                task="QA",
                generator="human/best",
                query=q,
                response=best_ans,
                label=0,
                grounding_evidence=grounding_evidence,
                stochastic_samples=[],
                claims=claims,
                metadata={"category": category, "pair_type": "factual", "q_idx": idx}
            ))

        # 2. Hallucinated / Misconception instance
        if incorrect_ans and incorrect_ans != "nan":
            claims = [
                ClaimItem(
                    claim_id=f"truthfulqa_{idx}_h_c{c_idx}",
                    text=c["text"],
                    label=1,
                    start_char=c["start_char"],
                    end_char=c["end_char"]
                )
                for c_idx, c in enumerate(split_sentences_clausal(incorrect_ans))
            ]
            records.append(DatasetRecord(
                id=f"truthfulqa_{idx}_hallu",
                group=group_id,
                dataset="TruthfulQA",
                task="QA",
                generator="human/misconception",
                query=q,
                response=incorrect_ans,
                label=1,
                grounding_evidence=grounding_evidence,
                stochastic_samples=[],
                claims=claims,
                metadata={"category": category, "pair_type": "hallucinated", "q_idx": idx}
            ))

    return records
