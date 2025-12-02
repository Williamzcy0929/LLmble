"""Type definitions and protocols for L-STAR package."""

from typing import TypedDict, Optional, List, Dict, Any
from pathlib import Path


class PairwiseResult(TypedDict):
    """Result of a single pairwise comparison."""
    ts: str
    model1_label: str
    model2_label: str
    gpt_choice: Optional[str]
    gpt_output: str
    repetition: int


class ConsensusResult(TypedDict):
    """Result of consensus clustering."""
    dataset: Optional[str]
    top_models: List[str]
    consensus_labels: List[int]
    consensus_ari: Optional[float]
    k_optimal: int
    k_ground_truth: Optional[int]
    k_determination: Dict[str, Any]

