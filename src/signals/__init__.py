"""
Signal Extraction Stack for HARMONIQ.
Includes:
- BaseSignal, SignalResult, SignalCache
- ConfidenceSignal (C)
- RetrievalSignal (R)
- SemanticSignal (S)
- SelfVerificationSignal (V)
"""

from src.signals.base import BaseSignal, SignalResult, SignalCache, compute_claim_hash
from src.signals.confidence import ConfidenceSignal
from src.signals.retrieval import RetrievalSignal
from src.signals.semantic import SemanticSignal
from src.signals.selfverif import SelfVerificationSignal

__all__ = [
    "BaseSignal",
    "SignalResult",
    "SignalCache",
    "compute_claim_hash",
    "ConfidenceSignal",
    "RetrievalSignal",
    "SemanticSignal",
    "SelfVerificationSignal"
]
