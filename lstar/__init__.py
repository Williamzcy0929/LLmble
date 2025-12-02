"""
L-STAR: LLM-guided Spatial Transcriptomics Analysis and Ranking

A pipeline for pairwise LLM-based model comparison and consensus clustering
for spatial transcriptomics data.
"""

from lstar.pairwise import run_pairwise_comparisons
from lstar.consensus import run_consensus_clustering
from lstar.pipeline import l_star

__version__ = "0.1.0"
__all__ = [
    "run_pairwise_comparisons",
    "run_consensus_clustering",
    "l_star",
]

