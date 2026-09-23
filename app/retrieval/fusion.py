"""Reciprocal Rank Fusion (RRF) and hybrid ranking algorithms."""

from typing import Dict, List
from app.retrieval.dense import RetrievalResult


class ReciprocalRankFusion:
    """Combines multiple ranked result lists using Reciprocal Rank Fusion."""

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, ranked_lists: List[List[RetrievalResult]], top_k: int = 10) -> List[RetrievalResult]:
        """Combine and re-score items across multiple ranked lists.

        Formula: RRF_score(d) = sum_{m in M} 1 / (k + rank_m(d))
        """
        scores: Dict[str, float] = {}
        items: Dict[str, RetrievalResult] = {}

        for ranked_list in ranked_lists:
            for rank, item in enumerate(ranked_list, start=1):
                chunk_id = item.chunk_id
                items[chunk_id] = item
                scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (self.k + rank))

        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        fused_results: List[RetrievalResult] = []
        for chunk_id, fused_score in sorted_chunks:
            original = items[chunk_id]
            fused_results.append(
                RetrievalResult(
                    chunk_id=original.chunk_id,
                    text=original.text,
                    score=fused_score,
                    metadata=original.metadata,
                )
            )

        return fused_results
