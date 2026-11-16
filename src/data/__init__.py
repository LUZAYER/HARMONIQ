"""
Dataset Loaders and Partitioning for Hallucination Benchmarks.
Strictly enforces group-disjoint splits to prevent question/source leakage.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class ClaimItem:
    claim_id: str
    text: str
    label: int  # 0 = factual, 1 = hallucinated
    start_char: int = -1
    end_char: int = -1
    entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetRecord:
    id: str
    group: str
    dataset: str
    task: str
    generator: str
    query: str
    response: str
    label: int  # 0 = factual, 1 = hallucinated
    grounding_evidence: str = ""
    stochastic_samples: List[str] = field(default_factory=list)
    claims: List[ClaimItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_claims(self) -> int:
        return len(self.claims)
