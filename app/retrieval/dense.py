"""Dense vector retrieval using ChromaDB vector store.

This module provides:
- :class:`RetrievalResult` — the canonical result schema shared across all
  retrieval paths (dense, sparse, hybrid).
- :class:`DenseRetriever` — ChromaDB kNN search, returning
  :class:`RetrievalResult` objects with populated ``chunk_id``, ``text``,
  ``score``, ``rank``, and ``metadata``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.config import get_settings

logger = logging.getLogger(__name__)


class RetrievalResult(BaseModel):
    """Schema for individual retrieved candidate items.

    Shared by dense (ChromaDB) and sparse (BM25) retrieval paths so that
    RRF fusion can operate over a single unified result type.
    """

    chunk_id: str = Field(..., description="ID of retrieved chunk")
    document_id: str = Field(default="", description="Parent document ID")
    text: str = Field(..., description="Text content of chunk")
    score: float = Field(..., description="Similarity or relevance score")
    rank: int = Field(default=0, description="1-indexed rank within the result list (0 = unranked)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata associated with chunk")


class DenseRetriever:
    """Manages dense semantic search over the ChromaDB vector store.

    The retriever is designed to be used with pre-computed embeddings: the
    caller supplies a ``query_vector`` (generated externally by
    :class:`~app.embeddings.embeddings.EmbeddingGenerator`) and this class
    performs the kNN search against the persistent ChromaDB collection.

    Args:
        collection_name: ChromaDB collection to query.
            Defaults to ``Settings.chroma_collection_name``.
        persist_directory: Path to the ChromaDB persistence directory.
            Defaults to ``Settings.chroma_persist_directory``.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_directory = persist_directory or settings.chroma_persist_directory
        self._store = None  # lazily initialised

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_store(self) -> None:
        """Initialise the VectorStore if not already done."""
        if self._store is None:
            from app.retrieval.vector_store import VectorStore

            self._store = VectorStore(
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query_vector: List[float], top_k: int = 10) -> List[RetrievalResult]:
        """Perform dense vector kNN search against ChromaDB.

        Args:
            query_vector: Pre-computed embedding of the query (list of floats).
            top_k: Maximum number of results to return.

        Returns:
            Ordered list of :class:`RetrievalResult` objects, best-first
            (highest cosine similarity first).  Returns an empty list when
            the collection is empty or ``top_k`` is zero/negative.
        """
        if top_k <= 0:
            return []

        self._ensure_store()

        try:
            raw = self._store._collection.query(
                query_embeddings=[query_vector],
                n_results=min(top_k, self._store.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error("ChromaDB query failed: %s", exc)
            return []

        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        results: List[RetrievalResult] = []
        for rank_idx, (chunk_id, text, meta, dist) in enumerate(
            zip(ids, documents, metadatas, distances), start=1
        ):
            # ChromaDB returns L2 or cosine *distance* (lower = more similar).
            # Convert distance to a similarity score in [0, 1] where 1 = perfect match.
            score = 1.0 / (1.0 + dist) if dist >= 0 else 0.0

            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    document_id=meta.get("document_id", ""),
                    text=text or "",
                    score=score,
                    rank=rank_idx,
                    metadata=meta or {},
                )
            )

        logger.debug(
            "DenseRetriever: query → %d result(s) (top score=%.4f)",
            len(results),
            results[0].score if results else 0.0,
        )
        return results
