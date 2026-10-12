"""Cross-encoder reranking module.

Provides :class:`CrossEncoderReranker` which loads a ``sentence-transformers``
cross-encoder model **lazily** (on first call to :meth:`rerank`) so that merely
importing this module or the ``app`` package does not trigger a model download.

The cross-encoder scores each (query, document-text) pair independently and
reorders the candidates from highest to lowest predicted relevance.

Design notes
------------
- Model loading is deferred to the first :meth:`rerank` call.
- An import-time or load-time failure is stored and re-raised on every
  subsequent call — the reranker will *not* silently fall back to the
  original ordering.  Callers that want a graceful degradation should
  catch :exc:`CrossEncoderError` and use the RRF-fused list directly.
- Empty candidate lists and ``top_k <= 0`` are handled safely.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from app.config import get_settings
from app.retrieval.dense import RetrievalResult

logger = logging.getLogger(__name__)


class CrossEncoderError(Exception):
    """Raised when the cross-encoder model cannot be loaded or scored."""


class CrossEncoderReranker:
    """Reranks candidate chunks against a query using a cross-encoder model.

    Args:
        model_name: ``sentence-transformers`` cross-encoder model identifier.
            Defaults to ``Settings.cross_encoder_model``.

    The underlying model is loaded on the *first* call to :meth:`rerank`.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        settings = get_settings()
        self.model_name: str = model_name or settings.cross_encoder_model
        self._model = None           # populated lazily
        self._load_error: Optional[Exception] = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Load the cross-encoder model if it has not been loaded yet.

        Raises:
            CrossEncoderError: The ``sentence-transformers`` package is not
                installed, or the model identifier is invalid / unreachable.
        """
        if self._model is not None:
            return

        if self._load_error is not None:
            raise CrossEncoderError(
                f"Cross-encoder model '{self.model_name}' failed to load: "
                f"{self._load_error}"
            ) from self._load_error

        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import]

            logger.info("Loading cross-encoder model '%s' …", self.model_name)
            self._model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder model '%s' loaded successfully.", self.model_name)
        except ImportError as exc:
            self._load_error = exc
            raise CrossEncoderError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc
        except Exception as exc:
            self._load_error = exc
            raise CrossEncoderError(
                f"Failed to load cross-encoder model '{self.model_name}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Rerank *candidates* using cross-encoder scoring against *query*.

        Each (query, candidate-text) pair is scored by the cross-encoder.
        Candidates are reordered from highest to lowest predicted relevance,
        and at most *top_k* results are returned.

        Args:
            query: The user's search query.
            candidates: Ordered list of :class:`RetrievalResult` objects to
                rerank.  Typically the output of RRF fusion.
            top_k: Maximum number of results to return.  Clamped to
                ``max(1, top_k)`` if positive, or returns all candidates
                when ``top_k <= 0``.

        Returns:
            Reranked list of :class:`RetrievalResult` objects with the
            ``score`` field updated to the cross-encoder relevance score.

        Raises:
            CrossEncoderError: The model cannot be loaded or inference fails.
                Callers should catch this and fall back to the RRF ordering
                if graceful degradation is required.
        """
        if not candidates:
            return []

        effective_k = max(1, top_k) if top_k > 0 else len(candidates)

        # Load model (raises CrossEncoderError on failure — never falls back silently)
        self._ensure_model()

        # Build (query, text) pairs for the cross-encoder
        pairs = [(query, result.text) for result in candidates]

        try:
            scores = self._model.predict(pairs).tolist()
        except Exception as exc:
            raise CrossEncoderError(
                f"Cross-encoder inference failed: {exc}"
            ) from exc

        # Attach scores and sort descending
        scored = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )

        reranked: List[RetrievalResult] = []
        for ce_score, result in scored[:effective_k]:
            reranked.append(
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    text=result.text,
                    score=float(ce_score),
                    rank=len(reranked) + 1,
                    metadata=result.metadata,
                )
            )

        logger.debug(
            "Cross-encoder reranking '%s': %d → %d result(s) (top score=%.4f)",
            query,
            len(candidates),
            len(reranked),
            reranked[0].score if reranked else 0.0,
        )
        return reranked

    @property
    def is_loaded(self) -> bool:
        """Return *True* if the model has been loaded into memory."""
        return self._model is not None
