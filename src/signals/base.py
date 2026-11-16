"""
Base Interface and Persistent Parquet Caching for Verification Signals.

Every signal returns:
- score in [0.0, 1.0] where 1.0 indicates high likelihood of hallucination
- available (bool): True if signal could be computed; False if inputs missing (returns NaN, NEVER 0.5)
- latency_ms (float)
"""

import os
import hashlib
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd


@dataclass
class SignalResult:
    score: float       # In [0.0, 1.0], or np.nan if unavailable
    available: bool    # True if computed, False if unavailable/skipped
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if not self.available:
            self.score = np.nan
        elif not np.isnan(self.score):
            self.score = float(np.clip(self.score, 0.0, 1.0))
        if self.metadata is None:
            self.metadata = {}


def compute_claim_hash(claim_text: str, context_text: str = "", model_version: str = "v1") -> str:
    """
    Computes a deterministic MD5 hash for caching signal outputs.
    """
    payload = f"{claim_text.strip()}|||{context_text.strip()}|||{model_version}".encode('utf-8')
    return hashlib.sha256(payload).hexdigest()[:24]


class SignalCache:
    """
    Persistent parquet-based key-value cache for claim signal evaluations.
    """
    def __init__(self, cache_dir: str = "data/cache/signals", signal_name: str = "base"):
        self.cache_dir = cache_dir
        self.signal_name = signal_name
        self.cache_file = os.path.join(cache_dir, f"{signal_name}.parquet")
        os.makedirs(cache_dir, exist_ok=True)
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.cache_file):
            try:
                df = pd.read_parquet(self.cache_file)
                for _, row in df.iterrows():
                    self.memory_cache[row['claim_hash']] = {
                        'score': row['score'],
                        'available': bool(row['available']),
                        'latency_ms': row.get('latency_ms', 0.0)
                    }
            except Exception as e:
                print(f"Warning: Failed to load signal cache from {self.cache_file}: {e}")

    def get(self, claim_hash: str) -> Optional[SignalResult]:
        if claim_hash in self.memory_cache:
            entry = self.memory_cache[claim_hash]
            return SignalResult(
                score=entry['score'],
                available=entry['available'],
                latency_ms=entry.get('latency_ms', 0.0),
                metadata={"from_cache": True}
            )
        return None

    def put(self, claim_hash: str, result: SignalResult):
        self.memory_cache[claim_hash] = {
            'score': result.score,
            'available': result.available,
            'latency_ms': result.latency_ms
        }

    def flush(self):
        if not self.memory_cache:
            return
        rows = []
        for ch, v in self.memory_cache.items():
            rows.append({
                'claim_hash': ch,
                'score': v['score'],
                'available': v['available'],
                'latency_ms': v.get('latency_ms', 0.0)
            })
        df = pd.DataFrame(rows)
        df.to_parquet(self.cache_file, index=False)


class BaseSignal:
    """Abstract base class for all HARMONIQ verification signals."""

    def __init__(self, name: str, version: str = "v1.0", cache_dir: str = "data/cache/signals"):
        self.name = name
        self.version = version
        self.cache = SignalCache(cache_dir=cache_dir, signal_name=name)

    def evaluate_claim(
        self,
        claim_text: str,
        evidence_text: str = "",
        query_text: str = "",
        samples: Optional[List[str]] = None
    ) -> SignalResult:
        """
        Evaluate a single claim, utilizing disk cache if available.
        """
        ch = compute_claim_hash(claim_text, evidence_text, self.version)
        cached = self.cache.get(ch)
        if cached is not None:
            return cached

        res = self._compute(claim_text, evidence_text, query_text, samples)
        self.cache.put(ch, res)
        return res

    def _compute(
        self,
        claim_text: str,
        evidence_text: str,
        query_text: str,
        samples: Optional[List[str]]
    ) -> SignalResult:
        raise NotImplementedError
