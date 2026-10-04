"""Chunk indexer: coordinates embedding generation and ChromaDB storage.

This module ties the :class:`~app.embeddings.embeddings.EmbeddingGenerator`
and :class:`~app.retrieval.vector_store.VectorStore` together into a single
:class:`ChunkIndexer` that the CLI scripts can call.

Responsibilities:
- Accept a list of :class:`~app.chunking.base.Chunk` objects.
- Extract metadata in the format expected by :class:`VectorStore`.
- Batch-embed the chunk texts via :class:`EmbeddingGenerator`.
- Upsert chunk records into ChromaDB (idempotent).
- Report progress and statistics.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from app.chunking.base import Chunk
from app.embeddings.embeddings import EmbeddingGenerator
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class ChunkIndexer:
    """Coordinates embedding and ChromaDB indexing for a list of chunks.

    Args:
        embedding_generator: Pre-configured :class:`EmbeddingGenerator`.
        vector_store: Pre-configured :class:`VectorStore`.
    """

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_store: VectorStore,
    ) -> None:
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_chunks(self, chunks: List[Chunk]) -> int:
        """Embed and upsert *chunks* into ChromaDB.

        Processing is done in batches controlled by
        :attr:`EmbeddingGenerator.batch_size`.  Chunks that fail to embed are
        logged and skipped; the method returns the number of successfully
        indexed chunks.

        Args:
            chunks: List of :class:`~app.chunking.base.Chunk` objects.

        Returns:
            Number of chunks successfully indexed.

        Raises:
            Nothing unrecoverable – failures per batch are logged and counted.
        """
        if not chunks:
            logger.warning("index_chunks called with an empty list; nothing to do.")
            return 0

        batch_size = self.embedding_generator.batch_size
        total_indexed = 0
        total_chunks = len(chunks)

        logger.info(
            "Indexing %d chunk(s) in batches of %d …",
            total_chunks,
            batch_size,
        )

        for batch_start in range(0, total_chunks, batch_size):
            batch = chunks[batch_start : batch_start + batch_size]
            batch_end = batch_start + len(batch)

            # Extract text for embedding
            texts = [c.text for c in batch]

            try:
                embeddings = self.embedding_generator.embed_texts(texts)
            except Exception as exc:
                logger.error(
                    "Embedding failed for batch %d-%d: %s — skipping batch.",
                    batch_start,
                    batch_end,
                    exc,
                )
                continue

            if len(embeddings) != len(batch):
                logger.error(
                    "Embedding count mismatch (expected %d, got %d) for batch %d-%d — skipping.",
                    len(batch),
                    len(embeddings),
                    batch_start,
                    batch_end,
                )
                continue

            ids = [c.chunk_id for c in batch]
            documents = texts
            metadatas = [_chunk_to_metadata(c) for c in batch]

            try:
                self.vector_store.upsert_chunks(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
                total_indexed += len(batch)
                logger.info(
                    "Indexed batch %d-%d (%d chunk(s)) — running total: %d",
                    batch_start,
                    batch_end,
                    len(batch),
                    total_indexed,
                )
            except Exception as exc:
                logger.error(
                    "ChromaDB upsert failed for batch %d-%d: %s — skipping batch.",
                    batch_start,
                    batch_end,
                    exc,
                )

        logger.info(
            "Indexing complete: %d / %d chunk(s) indexed successfully.",
            total_indexed,
            total_chunks,
        )
        return total_indexed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_to_metadata(chunk: Chunk) -> dict:
    """Convert a :class:`Chunk` to a flat ChromaDB-compatible metadata dict.

    Only scalar values are included.  The ``metadata`` sub-dict on the chunk
    is merged in, with top-level chunk fields taking precedence.

    Args:
        chunk: Source chunk object.

    Returns:
        Flat dict with scalar values only.
    """
    meta: dict = {}

    # Merge any extra metadata from the chunk's own metadata dict first
    if chunk.metadata:
        meta.update({k: v for k, v in chunk.metadata.items() if v is not None})

    # Top-level chunk fields always take precedence
    meta["document_id"] = chunk.document_id
    meta["chunk_id"] = chunk.chunk_id
    meta["source_file"] = chunk.source_file or ""
    meta["chunk_index"] = chunk.chunk_index
    meta["start_char"] = chunk.start_char
    meta["end_char"] = chunk.end_char
    meta["character_count"] = chunk.char_count
    meta["chunking_strategy"] = chunk.strategy

    if chunk.page_number is not None:
        meta["page_number"] = chunk.page_number
    if chunk.section_heading is not None:
        meta["section_heading"] = chunk.section_heading

    # file_type comes from the chunk.metadata dict (set during chunking)
    return meta
