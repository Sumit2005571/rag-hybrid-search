"""Base interfaces and data structures for document chunking."""

import hashlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Container for a single chunk of text and its metadata.

    All fields except ``text`` and ``chunk_index`` have defaults so that the
    model remains constructable from legacy code that only supplies the minimum
    required fields.
    """

    # --- identifiers ---
    chunk_id: str = Field(..., description="Deterministic unique identifier for the chunk")
    document_id: str = Field(..., description="Parent document identifier (from ingestion)")
    source_file: str = Field(default="", description="Absolute path of the originating file")

    # --- position ---
    chunk_index: int = Field(..., description="0-indexed position of this chunk within the document")
    start_char: int = Field(default=0, description="Start character offset within the document text")
    end_char: int = Field(default=0, description="End character offset within the document text")

    # --- content ---
    text: str = Field(..., description="Chunk text content")
    char_count: int = Field(default=0, description="Number of characters in this chunk")

    # --- provenance from source document ---
    section_heading: Optional[str] = Field(
        default=None,
        description="Section heading from the source document, if available",
    )
    page_number: Optional[int] = Field(
        default=None,
        description="Source page number (1-indexed), if available",
    )

    # --- strategy ---
    strategy: str = Field(
        default="fixed",
        description="Name of the chunking strategy that produced this chunk: 'fixed', 'recursive', or 'semantic'",
    )

    # --- arbitrary extra metadata ---
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata tags")

    def model_post_init(self, __context: Any) -> None:
        """Auto-populate char_count and end_char if not explicitly set."""
        if not self.char_count:
            object.__setattr__(self, "char_count", len(self.text))
        if not self.end_char and self.text:
            object.__setattr__(self, "end_char", self.start_char + len(self.text))


def generate_chunk_id(document_id: str, chunk_index: int, strategy: str) -> str:
    """Generate a deterministic chunk ID.

    The ID encodes the document ID, strategy prefix, and chunk index so that
    the same input always produces the same identifier (idempotent chunking).

    Args:
        document_id: Parent document's deterministic ID.
        chunk_index: 0-indexed position of the chunk within the document.
        strategy: One of ``'fixed'``, ``'recursive'``, or ``'semantic'``.

    Returns:
        A stable string identifier, e.g. ``"abc123#c5"`` for fixed strategy.
    """
    prefix_map = {"fixed": "c", "recursive": "r", "semantic": "s"}
    prefix = prefix_map.get(strategy, "c")
    return f"{document_id}#{prefix}{chunk_index}"


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies.

    Subclasses must implement :meth:`chunk` and set :attr:`strategy_name`.
    """

    #: Short identifier written into each produced :class:`Chunk`.
    strategy_name: str = "base"

    @abstractmethod
    def chunk(
        self,
        text: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Split *text* into a list of :class:`Chunk` objects.

        Args:
            text: The full document text to split.
            document_id: Stable identifier for the parent document.
            metadata: Optional dictionary of document-level metadata to
                propagate into each chunk (e.g. ``source_file``,
                ``section_heading``, ``page_number``).

        Returns:
            Ordered list of :class:`Chunk` objects with deterministic IDs.
        """
        ...
