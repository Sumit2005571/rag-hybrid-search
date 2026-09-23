"""Cross-encoder reranking module."""

from typing import List, Optional
from app.config import get_settings
from app.retrieval.dense import RetrievalResult


class CrossEncoderReranker:
    """Reranks candidate chunks against a query using a cross-encoder model."""

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.cross_encoder_model

    def rerank(self, query: str, candidates: List[RetrievalResult], top_k: int = 5) -> List[RetrievalResult]:
        """Rerank retrieved candidates based on cross-encoder scoring.

        Note: Cross-encoder inference will be wired in the reranker milestone.
        """
        # Return top_k candidates directly as fallback for foundation milestone
        return candidates[:top_k]
