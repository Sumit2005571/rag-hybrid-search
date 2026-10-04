"""BM25 sparse retrieval store for the RAG pipeline.

This module provides :class:`BM25SparseStore` — a complete, persistent BM25
index built over the same canonical :class:`~app.chunking.base.Chunk` objects
used by the ChromaDB dense index.

Key features
------------
- **Same canonical chunks**: indexes the identical ``Chunk`` objects used by
  the ChromaDB pipeline; chunk IDs, metadata, and text are shared.
- **Deterministic tokenization**: :func:`tokenize` preserves technical
  identifiers (``DATABASE_URL``, ``get_user_by_id``, ``HTTP_500``) while
  normalising punctuation and whitespace.
- **Persistent index**: saved as a structured JSON + pickle pair under
  ``data/bm25/`` so the index survives restarts without re-indexing.
- **Idempotent rebuild**: calling :meth:`build_index` with the same canonical
  chunks produces the same mapping and index; running it twice does not add
  duplicates.
- **Unified result format**: returns :class:`~app.retrieval.dense.RetrievalResult`
  objects (shared with the dense path) so Phase 2.3 RRF can operate without
  conversion.

Design note — full rebuild strategy
-------------------------------------
BM25Okapi stores global corpus statistics (IDF) that change whenever a chunk
is added or removed.  Incremental updates therefore require recomputing IDF
over the whole corpus anyway.  This implementation performs a **full
deterministic rebuild** on every :meth:`build_index` call, which is simpler,
more reliable, and safe to run repeatedly.  The rebuild is efficient for
corpora of typical RAG sizes (thousands to tens-of-thousands of chunks).
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.chunking.base import Chunk
from app.retrieval.dense import RetrievalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INDEX_META_FILE = "bm25_meta.json"
_INDEX_MODEL_FILE = "bm25_model.pkl"
_SCHEMA_VERSION = "1.0"

_DEFAULT_INDEX_DIR = "./data/bm25"
_DEFAULT_TOP_K = 10


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

# Regex that splits on whitespace OR on transitions between alphanumeric runs
# and pure-punctuation runs, while preserving underscores (important for
# technical identifiers like DATABASE_URL, get_user_by_id, MAX_RETRIES).
#
# Strategy:
#   1. Lowercase the text.
#   2. Replace runs of whitespace with a single space.
#   3. Split on spaces and additional punctuation boundaries, but keep
#      underscores attached so DATABASE_URL stays as one token.
#   4. Filter out empty / whitespace-only tokens.
#   5. Keep numbers, hyphens within words, and dots within version strings.

_SPLIT_RE = re.compile(
    r"[^\w\-\.]+|(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])",
    re.UNICODE,
)


def tokenize(text: str) -> List[str]:
    """Deterministic, technically-aware tokenizer for BM25.

    Splits text into lowercase tokens while preserving technical identifiers:

    - ``DATABASE_URL``   → ``["database_url"]``
    - ``get_user_by_id`` → ``["get_user_by_id"]``
    - ``HTTP_500``        → ``["http_500"]``
    - ``MAX_RETRIES``     → ``["max_retries"]``
    - ``v1.2.3``          → ``["v1.2.3"]``
    - Camel-case names split at boundaries: ``getUserById``
      → ``["get", "user", "by", "id"]``

    Punctuation that is not part of a technical token is used as a delimiter.
    Numbers are kept as part of their containing token.

    Args:
        text: Raw text string to tokenize.

    Returns:
        Ordered list of lowercase token strings.  Empty strings are excluded.
        Returns an empty list for empty or whitespace-only input.
    """
    if not text or not text.strip():
        return []

    # Split using the regex (handles camelCase transitions too)
    # Must split BEFORE lowercasing so the lookarounds match uppercase letters
    raw_tokens = _SPLIT_RE.split(text)

    # Filter and clean, then lowercase
    tokens = [t.lower().strip(".-") for t in raw_tokens if t and t.strip()]
    # Remove tokens that are purely punctuation after stripping
    tokens = [t for t in tokens if re.search(r"[\w]", t)]
    return tokens


# ---------------------------------------------------------------------------
# Metadata helper
# ---------------------------------------------------------------------------

def _chunk_to_sparse_metadata(chunk: Chunk) -> Dict[str, Any]:
    """Build a flat metadata dict from a :class:`Chunk`.

    Only serialization-safe scalar types (str, int, float, bool) are
    included.  ``None`` values are coerced to ``""`` so the metadata can
    safely round-trip through JSON.

    Args:
        chunk: Source chunk object.

    Returns:
        Flat dict compatible with JSON serialisation.
    """
    meta: Dict[str, Any] = {}

    # Merge extra metadata from chunk's own dict first
    if chunk.metadata:
        for k, v in chunk.metadata.items():
            if v is None:
                meta[k] = ""
            elif isinstance(v, (str, int, float, bool)):
                meta[k] = v
            else:
                meta[k] = str(v)

    # Top-level fields (override extras)
    meta["document_id"] = chunk.document_id
    meta["chunk_id"] = chunk.chunk_id
    meta["source_file"] = chunk.source_file or ""
    meta["chunk_index"] = chunk.chunk_index
    meta["character_count"] = chunk.char_count
    meta["chunking_strategy"] = chunk.strategy
    meta["page_number"] = chunk.page_number if chunk.page_number is not None else ""
    meta["section_heading"] = chunk.section_heading if chunk.section_heading is not None else ""

    return meta


# ---------------------------------------------------------------------------
# BM25SparseStore
# ---------------------------------------------------------------------------

class BM25IndexError(Exception):
    """Raised when the BM25 index cannot be loaded or built."""


class BM25SparseStore:
    """Persistent BM25 sparse retrieval index over canonical chunks.

    The store maintains a :class:`rank_bm25.BM25Okapi` model plus auxiliary
    data structures needed to map BM25 results back to chunk IDs and metadata.

    Args:
        index_directory: Directory where the persisted index is stored.
            Created automatically if it does not exist.
        top_k: Default number of results returned by :meth:`search`.
    """

    def __init__(
        self,
        index_directory: Optional[str] = None,
        top_k: int = _DEFAULT_TOP_K,
    ) -> None:
        self.index_directory: Path = Path(
            index_directory
            or os.getenv("BM25_INDEX_DIRECTORY", _DEFAULT_INDEX_DIR)
        )
        self.top_k: int = max(1, top_k)

        # Internal state — populated by build_index() or load_index()
        self._bm25 = None                          # BM25Okapi instance
        self._chunk_ids: List[str] = []            # parallel to BM25 corpus rows
        self._chunk_texts: List[str] = []          # raw texts (for result construction)
        self._chunk_metadata: List[Dict[str, Any]] = []  # per-chunk metadata

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(self, chunks: List[Chunk]) -> int:
        """Build and persist the BM25 index from canonical chunks.

        The index is rebuilt deterministically from the provided *chunks* list.
        Calling this method twice with the same chunks produces the same index —
        no duplicate entries are created.

        Args:
            chunks: Ordered list of canonical :class:`~app.chunking.base.Chunk`
                objects (same objects used by ChromaDB).

        Returns:
            Number of chunks indexed.

        Raises:
            BM25IndexError: The index directory cannot be created or the
                index cannot be saved.
            ValueError: *chunks* is empty.
        """
        if not chunks:
            logger.warning("build_index called with empty chunks list — no index built.")
            return 0

        logger.info("Building BM25 index for %d chunk(s) …", len(chunks))

        # Validate and deduplicate by chunk_id (preserve first occurrence order)
        seen_ids: set = set()
        deduped: List[Chunk] = []
        for c in chunks:
            if c.chunk_id in seen_ids:
                logger.warning("Duplicate chunk_id '%s' — keeping first occurrence.", c.chunk_id)
            else:
                seen_ids.add(c.chunk_id)
                deduped.append(c)

        # Build corpus structures
        chunk_ids = [c.chunk_id for c in deduped]
        chunk_texts = [c.text for c in deduped]
        chunk_metas = [_chunk_to_sparse_metadata(c) for c in deduped]
        tokenized = [tokenize(t) for t in chunk_texts]

        # BM25Okapi requires at least one token in the corpus
        # Insert a sentinel for any completely empty chunk so the corpus remains valid
        tokenized_safe = [toks if toks else ["__empty__"] for toks in tokenized]

        from rank_bm25 import BM25Okapi

        bm25 = BM25Okapi(tokenized_safe)

        # Update internal state atomically
        self._bm25 = bm25
        self._chunk_ids = chunk_ids
        self._chunk_texts = chunk_texts
        self._chunk_metadata = chunk_metas

        # Persist
        self._save_index(tokenized_safe)

        logger.info(
            "BM25 index built and saved: %d chunk(s) -> %s",
            len(deduped),
            self.index_directory,
        )
        return len(deduped)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_index(self, tokenized_corpus: List[List[str]]) -> None:
        """Persist the BM25 model and metadata to *index_directory*.

        Two files are written:
        - ``bm25_meta.json`` — human-readable JSON containing chunk IDs,
          texts, and metadata.
        - ``bm25_model.pkl``— Python pickle of the BM25Okapi object and the
          tokenized corpus.

        The pickle contains only objects built from the project's own code
        and ``rank_bm25``.  It must only be loaded from this module's
        :meth:`load_index` method, which enforces the expected schema version.

        Args:
            tokenized_corpus: List of token lists (parallel to chunk arrays).

        Raises:
            BM25IndexError: The directory cannot be created or files cannot be
                written.
        """
        try:
            self.index_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise BM25IndexError(
                f"Cannot create BM25 index directory '{self.index_directory}': {exc}"
            ) from exc

        # JSON metadata (human-readable, schema-versioned)
        meta_payload = {
            "schema_version": _SCHEMA_VERSION,
            "chunk_count": len(self._chunk_ids),
            "chunk_ids": self._chunk_ids,
            "chunk_texts": self._chunk_texts,
            "chunk_metadata": self._chunk_metadata,
        }
        meta_path = self.index_directory / _INDEX_META_FILE
        try:
            meta_path.write_text(
                json.dumps(meta_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            raise BM25IndexError(f"Cannot write BM25 metadata to '{meta_path}': {exc}") from exc

        # Pickle the BM25 model + tokenized corpus
        model_path = self.index_directory / _INDEX_MODEL_FILE
        try:
            with model_path.open("wb") as fh:
                pickle.dump(
                    {
                        "schema_version": _SCHEMA_VERSION,
                        "bm25": self._bm25,
                        "tokenized_corpus": tokenized_corpus,
                    },
                    fh,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        except (OSError, pickle.PicklingError) as exc:
            raise BM25IndexError(f"Cannot write BM25 model to '{model_path}': {exc}") from exc

        logger.debug("BM25 index saved: %s", self.index_directory)

    def load_index(self) -> int:
        """Load the BM25 index from *index_directory*.

        Reads ``bm25_meta.json`` (chunk IDs, texts, metadata) and
        ``bm25_model.pkl`` (BM25Okapi object) from the index directory.

        Returns:
            Number of chunks loaded from the persisted index.

        Raises:
            BM25IndexError: Index files are missing, corrupted, or have an
                incompatible schema version.
        """
        meta_path = self.index_directory / _INDEX_META_FILE
        model_path = self.index_directory / _INDEX_MODEL_FILE

        if not meta_path.exists() or not model_path.exists():
            raise BM25IndexError(
                f"BM25 index not found in '{self.index_directory}'. "
                "Run 'python scripts/index_bm25.py' to build it."
            )

        # Load JSON metadata
        try:
            meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BM25IndexError(f"Cannot read BM25 metadata from '{meta_path}': {exc}") from exc

        schema_ver = meta_payload.get("schema_version", "unknown")
        if schema_ver != _SCHEMA_VERSION:
            raise BM25IndexError(
                f"BM25 index schema version mismatch: "
                f"expected '{_SCHEMA_VERSION}', found '{schema_ver}'. "
                "Please rebuild the index."
            )

        # Load BM25 model from pickle (project-generated files only)
        try:
            with model_path.open("rb") as fh:
                model_payload = pickle.load(fh)  # noqa: S301 — project-generated only
        except (OSError, pickle.UnpicklingError, Exception) as exc:
            raise BM25IndexError(f"Cannot load BM25 model from '{model_path}': {exc}") from exc

        model_ver = model_payload.get("schema_version", "unknown")
        if model_ver != _SCHEMA_VERSION:
            raise BM25IndexError(
                f"BM25 model schema version mismatch: expected '{_SCHEMA_VERSION}', "
                f"found '{model_ver}'. Please rebuild the index."
            )

        self._chunk_ids = meta_payload["chunk_ids"]
        self._chunk_texts = meta_payload["chunk_texts"]
        self._chunk_metadata = meta_payload["chunk_metadata"]
        self._bm25 = model_payload["bm25"]

        logger.info(
            "BM25 index loaded: %d chunk(s) from '%s'",
            len(self._chunk_ids),
            self.index_directory,
        )
        return len(self._chunk_ids)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Search the BM25 index for *query*.

        Tokenizes the query, scores all corpus documents via BM25Okapi, and
        returns the top-*k* results in descending score order.  Results with
        equal BM25 scores are ordered by chunk_id (lexicographic) for
        determinism.

        Args:
            query: Raw query string.
            top_k: Maximum number of results to return.
                   Defaults to :attr:`top_k` (set at construction).

        Returns:
            Ordered list of :class:`~app.retrieval.dense.RetrievalResult`
            objects with ``rank`` populated (1-indexed).

        Raises:
            BM25IndexError: The index has not been built or loaded.
            ValueError: *query* is empty or whitespace-only.
        """
        if not query or not query.strip():
            raise ValueError("BM25 search query must not be empty.")

        if self._bm25 is None:
            raise BM25IndexError(
                "BM25 index is not loaded. Call build_index() or load_index() first."
            )

        k = top_k if top_k is not None else self.top_k
        k = max(1, k)

        query_tokens = tokenize(query)
        if not query_tokens:
            logger.warning("Query '%s' produced no tokens — returning empty results.", query)
            return []

        scores: List[float] = self._bm25.get_scores(query_tokens).tolist()

        # Pair (score, chunk_id, corpus_index) and sort
        scored = sorted(
            enumerate(scores),
            key=lambda x: (-x[1], self._chunk_ids[x[0]]),  # desc score, asc chunk_id for ties
        )

        results: List[RetrievalResult] = []
        for rank_idx, (corpus_idx, score) in enumerate(scored[:k], start=1):
            chunk_id = self._chunk_ids[corpus_idx]
            text = self._chunk_texts[corpus_idx]
            meta = self._chunk_metadata[corpus_idx]
            doc_id = meta.get("document_id", "")

            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    text=text,
                    score=float(score),
                    rank=rank_idx,
                    metadata=meta,
                )
            )

        logger.debug(
            "BM25 search '%s' → %d result(s) (top score=%.4f)",
            query,
            len(results),
            results[0].score if results else 0.0,
        )
        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Return *True* if the index has been built or loaded."""
        return self._bm25 is not None

    @property
    def chunk_count(self) -> int:
        """Return the number of chunks currently in the index."""
        return len(self._chunk_ids)

    def get_index_info(self) -> Dict[str, Any]:
        """Return a summary dict describing the current index state."""
        return {
            "index_directory": str(self.index_directory),
            "chunk_count": self.chunk_count,
            "is_loaded": self.is_loaded,
            "default_top_k": self.top_k,
        }
