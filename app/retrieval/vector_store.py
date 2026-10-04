"""ChromaDB vector-store abstraction for the RAG pipeline.

This module wraps a persistent ChromaDB instance and exposes high-level
operations needed for the indexing phase:

- Create or reopen a persistent collection.
- Upsert chunks together with their embeddings and metadata.
- Retrieve basic collection statistics (e.g. total chunk count).
- Safe handling of duplicate IDs (upsert semantics, no duplicates).

Environment / config:
    CHROMA_PERSIST_DIRECTORY   Local path where ChromaDB stores its data.
                               Default: ./data/chroma
    CHROMA_COLLECTION_NAME     ChromaDB collection to use.
                               Default: rag_documents
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Raised on unrecoverable ChromaDB / vector-store errors."""


class VectorStore:
    """Persistent ChromaDB vector store wrapper.

    Creates or reopens a collection under *persist_directory*.  All write
    operations use ``upsert`` so the same chunk ID can be indexed multiple
    times without creating duplicate records.

    Args:
        persist_directory: Directory where ChromaDB persists its data.
            Created automatically if it does not exist.
        collection_name: Name of the ChromaDB collection.
        embedding_function: Optional ChromaDB-compatible embedding function.
            When ``None`` (default) embeddings must be supplied explicitly to
            :meth:`upsert_chunks`.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_function: Optional[Any] = None,
    ) -> None:
        # Resolve config (env > explicit arg > project Settings)
        self.persist_directory = persist_directory or os.getenv(
            "CHROMA_PERSIST_DIRECTORY", "./data/chroma"
        )
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )

        # Fall back to project Settings if available
        if not persist_directory and not os.getenv("CHROMA_PERSIST_DIRECTORY"):
            try:
                from app.config import get_settings

                settings = get_settings()
                self.persist_directory = settings.chroma_persist_directory
                self.collection_name = collection_name or settings.chroma_collection_name
            except Exception:
                pass

        self._embedding_function = embedding_function
        self._client = None
        self._collection = None

        self._initialize()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Create the ChromaDB client and open/create the collection."""
        try:
            import chromadb

            persist_path = Path(self.persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(path=str(persist_path))
            logger.info(
                "ChromaDB PersistentClient opened at: %s",
                self.persist_directory,
            )

            kwargs: Dict[str, Any] = {"name": self.collection_name}
            if self._embedding_function is not None:
                kwargs["embedding_function"] = self._embedding_function

            self._collection = self._client.get_or_create_collection(**kwargs)
            logger.info(
                "Collection '%s' ready (count=%d)",
                self.collection_name,
                self._collection.count(),
            )
        except ImportError as exc:
            raise VectorStoreError(
                "chromadb package is not installed. Run: pip install chromadb"
            ) from exc
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to initialise ChromaDB at '{self.persist_directory}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Upsert a batch of chunks into the collection.

        Uses ChromaDB's ``upsert`` operation so repeated calls with the same
        IDs update existing records rather than creating duplicates.

        Args:
            ids: Deterministic chunk IDs (one per chunk).
            documents: The raw text of each chunk.
            embeddings: Pre-computed embedding vectors (one per chunk).
            metadatas: Metadata dicts to store alongside each chunk.

        Raises:
            VectorStoreError: ChromaDB client is not initialised or an
                API error occurs.
            ValueError: Lists have different lengths or are empty.
        """
        if not ids:
            raise ValueError("upsert_chunks called with empty ids list")
        if not (len(ids) == len(documents) == len(embeddings) == len(metadatas)):
            raise ValueError(
                f"Mismatched lengths: ids={len(ids)}, documents={len(documents)}, "
                f"embeddings={len(embeddings)}, metadatas={len(metadatas)}"
            )
        if self._collection is None:
            raise VectorStoreError("ChromaDB collection is not initialised.")

        # Sanitise metadata: ChromaDB requires scalar values (str/int/float/bool)
        sanitised = [_sanitise_metadata(m) for m in metadatas]

        try:
            self._collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=sanitised,
            )
            logger.debug("Upserted %d chunk(s) into '%s'.", len(ids), self.collection_name)
        except Exception as exc:
            raise VectorStoreError(
                f"ChromaDB upsert failed for {len(ids)} chunk(s): {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Read / statistics
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the total number of chunks in the collection."""
        if self._collection is None:
            raise VectorStoreError("ChromaDB collection is not initialised.")
        return self._collection.count()

    def get_collection_info(self) -> Dict[str, Any]:
        """Return a summary dict with collection name and chunk count."""
        return {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "chunk_count": self.count(),
        }

    def chunk_exists(self, chunk_id: str) -> bool:
        """Return *True* if *chunk_id* is already in the collection."""
        if self._collection is None:
            raise VectorStoreError("ChromaDB collection is not initialised.")
        result = self._collection.get(ids=[chunk_id])
        return len(result["ids"]) > 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitise_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Convert metadata values to ChromaDB-compatible scalar types.

    ChromaDB only accepts ``str``, ``int``, ``float``, and ``bool`` values in
    metadata dicts.  This helper converts ``None`` → ``""`` and coerces
    everything else to ``str`` so storage never fails silently.

    Args:
        meta: Raw metadata dict (may contain None or complex values).

    Returns:
        A new dict with only ChromaDB-compatible values.
    """
    sanitised: Dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            sanitised[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            sanitised[key] = value
        else:
            sanitised[key] = str(value)
    return sanitised
