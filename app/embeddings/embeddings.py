"""Embedding generation abstraction for RAG pipeline.

This module provides the :class:`EmbeddingGenerator` which wraps the OpenAI
text-embedding API (v3.x SDK style using ``openai.Embedding.create``).

Key responsibilities:
- Create / reuse the embedding client
- Generate embeddings for lists of texts (batch processing)
- Expose a configurable model name and batch size
- Raise clear errors when the API key is missing
- Avoid re-embedding the same content unnecessarily

Environment variables:
    OPENAI_API_KEY           Required for real embedding calls.
    OPENAI_EMBEDDING_MODEL   Default: text-embedding-3-small
    EMBEDDING_BATCH_SIZE     Default: 100
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

# Sensible default – can be overridden via config or environment
_DEFAULT_BATCH_SIZE = 100
_DEFAULT_MODEL = "text-embedding-3-small"


class EmbeddingConfigError(Exception):
    """Raised when the embedding client cannot be configured (e.g. missing API key)."""


class EmbeddingAPIError(Exception):
    """Raised when the embedding API call fails."""


class EmbeddingGenerator:
    """Generate dense vector embeddings via OpenAI's text-embedding models.

    Uses the OpenAI Python SDK v3.x (``openai.Embedding.create``).

    Args:
        model: Embedding model name.  Defaults to the value of
            ``OPENAI_EMBEDDING_MODEL`` env var, then ``text-embedding-3-small``.
        api_key: OpenAI API key.  Defaults to ``OPENAI_API_KEY`` env var.
        batch_size: Maximum number of texts sent in a single API request.
        validate_on_init: If *True* (default) raise
            :exc:`EmbeddingConfigError` immediately when the API key is absent.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        validate_on_init: bool = True,
    ) -> None:
        # Resolve model
        self.model = model or os.getenv("OPENAI_EMBEDDING_MODEL", _DEFAULT_MODEL)
        self.batch_size = max(1, batch_size)

        # Resolve API key (env var > explicit arg > Settings)
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            # Try via project Settings (lazy import to avoid circular deps)
            try:
                from app.config import get_settings

                settings = get_settings()
                resolved_key = settings.openai_api_key
            except Exception:
                pass

        self._api_key: Optional[str] = resolved_key

        if validate_on_init and not self._api_key:
            raise EmbeddingConfigError(
                "OPENAI_API_KEY is not set. "
                "Export the environment variable or add it to your .env file before running embeddings."
            )

        logger.debug(
            "EmbeddingGenerator initialised: model=%s, batch_size=%d",
            self.model,
            self.batch_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Texts are sent to the OpenAI API in batches of at most
        :attr:`batch_size` items.  The returned list preserves the input order.

        Args:
            texts: Non-empty list of text strings to embed.

        Returns:
            List of float vectors (one per input text).

        Raises:
            EmbeddingConfigError: API key is not set.
            EmbeddingAPIError: The OpenAI API returned an error.
            ValueError: *texts* is empty.
        """
        if not texts:
            return []

        self._ensure_api_key()

        all_embeddings: List[List[float]] = []
        total = len(texts)

        for batch_start in range(0, total, self.batch_size):
            batch = texts[batch_start : batch_start + self.batch_size]
            batch_end = batch_start + len(batch)
            logger.debug(
                "Embedding batch %d-%d / %d (model=%s)",
                batch_start,
                batch_end,
                total,
                self.model,
            )
            try:
                embeddings = self._call_api(batch)
            except EmbeddingConfigError:
                raise
            except Exception as exc:
                raise EmbeddingAPIError(
                    f"OpenAI embedding API call failed for batch {batch_start}-{batch_end}: {exc}"
                ) from exc

            all_embeddings.extend(embeddings)

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """Generate an embedding for a single query string.

        Args:
            query: Text to embed.

        Returns:
            A single float vector.
        """
        results = self.embed_texts([query])
        return results[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_api_key(self) -> None:
        """Re-check the API key right before an API call."""
        if not self._api_key:
            raise EmbeddingConfigError(
                "OPENAI_API_KEY is not set. "
                "Export the environment variable or add it to your .env file."
            )

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Call the OpenAI embedding API for a batch of texts.

        Uses the v3 SDK pattern: ``openai.Embedding.create``.

        Args:
            texts: A batch of text strings (already within batch_size limit).

        Returns:
            List of embedding vectors in the same order as *texts*.
        """
        import openai  # deferred to allow tests to mock

        openai.api_key = self._api_key
        response = openai.Embedding.create(model=self.model, input=texts)
        # Response structure: response["data"][i]["embedding"]
        sorted_data = sorted(response["data"], key=lambda d: d["index"])
        return [item["embedding"] for item in sorted_data]
