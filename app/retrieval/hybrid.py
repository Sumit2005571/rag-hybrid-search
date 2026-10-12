"""Hybrid retrieval service combining dense and sparse retrieval.

This module provides :class:`HybridRetriever` — the central orchestration
class for the retrieval phase of the RAG pipeline.

Retrieval flow
--------------
1. **Dense retrieval**: embed the query via
   :class:`~app.embeddings.embeddings.EmbeddingGenerator`, then perform a
   kNN search in ChromaDB using :class:`~app.retrieval.dense.DenseRetriever`.
2. **Sparse retrieval**: tokenise the query and search the pre-built BM25
   index using :class:`~app.retrieval.sparse.BM25SparseStore`.
3. **RRF fusion**: merge the two ranked lists with
   :class:`~app.retrieval.fusion.ReciprocalRankFusion`, deduplicating
   chunks that appear in both lists (the chunk with the better combined
   score wins).
4. **Cross-encoder reranking** *(optional)*: pass the fused candidates
   through :class:`~app.retrieval.reranker.CrossEncoderReranker` to
   produce a final, precisely-scored ranking.  If reranking fails,
   :exc:`~app.retrieval.reranker.CrossEncoderError` is re-raised so the
   caller can decide to fall back to the RRF list.

Design choices
--------------
- Dense and sparse retrieval are called sequentially (not concurrently) to
  keep the implementation simple and debuggable.
- The BM25 index must already be built / loaded before calling
  :meth:`retrieve`.  The :class:`HybridRetriever` does not rebuild the
  index automatically.
- Embedding generation requires a valid OpenAI API key at call time.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from app.config import get_settings
from app.retrieval.dense import DenseRetriever, RetrievalResult
from app.retrieval.fusion import ReciprocalRankFusion
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.sparse import BM25SparseStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Orchestrates dense + sparse retrieval, RRF fusion, and cross-encoder reranking.

    Args:
        dense_retriever: A configured :class:`DenseRetriever`.
        sparse_store: A loaded :class:`BM25SparseStore`.
        embedding_generator: An :class:`~app.embeddings.embeddings.EmbeddingGenerator`
            used to embed the query before dense retrieval.
        top_k_dense: Number of candidates from dense retrieval.
        top_k_sparse: Number of candidates from sparse retrieval.
        top_k_final: Maximum number of results returned after reranking.
        rrf_k: RRF constant *k* (typically 60).
        reranker: Optional :class:`CrossEncoderReranker`.  When *None*, the
            fused RRF list is returned directly (no cross-encoder scoring).
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_store: BM25SparseStore,
        embedding_generator,
        top_k_dense: Optional[int] = None,
        top_k_sparse: Optional[int] = None,
        top_k_final: Optional[int] = None,
        rrf_k: Optional[int] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> None:
        settings = get_settings()

        self.dense_retriever = dense_retriever
        self.sparse_store = sparse_store
        self.embedding_generator = embedding_generator

        self.top_k_dense: int = max(1, top_k_dense or settings.top_k_dense)
        self.top_k_sparse: int = max(1, top_k_sparse or settings.top_k_sparse)
        self.top_k_final: int = max(1, top_k_final or settings.top_k_final)
        self.rrf_k: int = max(1, rrf_k or settings.rrf_k)

        self._rrf = ReciprocalRankFusion(k=self.rrf_k)
        self.reranker = reranker

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_reranker: bool = True,
    ) -> List[RetrievalResult]:
        """Run the full hybrid retrieval pipeline for *query*.

        Args:
            query: Raw user query string.
            top_k: Override for the number of final results.  Defaults to
                ``self.top_k_final``.
            use_reranker: When *True* (default) and a :attr:`reranker` is
                configured, apply cross-encoder reranking.  Set to *False*
                to return the RRF-fused list without reranking.

        Returns:
            Ordered list of :class:`RetrievalResult` objects.  At most
            *top_k* (or ``self.top_k_final``) results are returned.

        Raises:
            ValueError: *query* is empty or whitespace-only.
            ``CrossEncoderError``: Reranker model failed to load or score.
                Only raised when *use_reranker* is *True* and a reranker
                is configured.
        """
        if not query or not query.strip():
            raise ValueError("Hybrid retrieval query must not be empty.")

        k_final = max(1, top_k) if top_k and top_k > 0 else self.top_k_final

        # --- Step 1: Dense retrieval -----------------------------------------
        dense_results = self._dense_search(query)
        logger.debug("Dense retrieval: %d result(s)", len(dense_results))

        # --- Step 2: Sparse retrieval ----------------------------------------
        sparse_results = self._sparse_search(query)
        logger.debug("Sparse retrieval: %d result(s)", len(sparse_results))

        # --- Step 3: RRF fusion ----------------------------------------------
        # Pass more candidates than k_final so the reranker has enough to pick from
        rrf_k = max(k_final, self.top_k_dense, self.top_k_sparse)
        fused = self._rrf.fuse([dense_results, sparse_results], top_k=rrf_k)
        logger.debug("RRF fusion: %d candidate(s)", len(fused))

        if not fused:
            return []

        # --- Step 4: Cross-encoder reranking (optional) ----------------------
        if use_reranker and self.reranker is not None:
            reranked = self.reranker.rerank(query, fused, top_k=k_final)
            logger.info(
                "Hybrid retrieval (with reranking): query='%s' → %d result(s)",
                query,
                len(reranked),
            )
            return reranked

        # No reranker — return RRF-fused list truncated to k_final
        final = fused[:k_final]
        logger.info(
            "Hybrid retrieval (RRF only): query='%s' → %d result(s)",
            query,
            len(final),
        )
        return final

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dense_search(self, query: str) -> List[RetrievalResult]:
        """Embed *query* and search ChromaDB.  Returns empty list on failure."""
        try:
            query_vector = self.embedding_generator.embed_query(query)
            return self.dense_retriever.search(query_vector, top_k=self.top_k_dense)
        except Exception as exc:
            logger.warning("Dense retrieval failed for query '%s': %s", query, exc)
            return []

    def _sparse_search(self, query: str) -> List[RetrievalResult]:
        """BM25 search.  Returns empty list on failure (index not loaded, etc.)."""
        try:
            return self.sparse_store.search(query, top_k=self.top_k_sparse)
        except Exception as exc:
            logger.warning("Sparse retrieval failed for query '%s': %s", query, exc)
            return []
