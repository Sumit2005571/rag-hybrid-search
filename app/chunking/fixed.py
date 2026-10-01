"""Fixed-size character chunking with configurable overlap.

Splits a document into windows of exactly *chunk_size* characters (or fewer
for the final window), advancing by ``chunk_size - chunk_overlap`` characters
between consecutive windows.

No text is duplicated or dropped: every character of the input appears in at
least one chunk, and the overlap region appears in exactly two consecutive
chunks.
"""

from typing import Any, Dict, List, Optional

from app.chunking.base import BaseChunker, Chunk, generate_chunk_id
from app.logging_config import get_logger

logger = get_logger(__name__)


class FixedSizeChunker(BaseChunker):
    """Chunk text into fixed-size character windows with configurable overlap.

    Args:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of characters that consecutive chunks share.
            Must be strictly less than *chunk_size*.

    Example::

        chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk(text=doc.content, document_id=doc.metadata.document_id)
    """

    strategy_name: str = "fixed"

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than "
                f"chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        text: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Split *text* into overlapping fixed-size windows.

        Args:
            text: Document text to split.
            document_id: Deterministic parent document ID.
            metadata: Document-level metadata propagated to each chunk.
                Recognised keys: ``source_file``, ``section_heading``,
                ``page_number``.

        Returns:
            Ordered list of :class:`~app.chunking.base.Chunk` objects.
        """
        if not text:
            return []

        meta = metadata or {}
        source_file = meta.get("source_file", "")
        section_heading = meta.get("section_heading")
        page_number = meta.get("page_number")

        chunks: List[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        idx = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            chunks.append(
                Chunk(
                    chunk_id=generate_chunk_id(document_id, idx, self.strategy_name),
                    document_id=document_id,
                    source_file=source_file,
                    chunk_index=idx,
                    start_char=start,
                    end_char=end,
                    text=chunk_text,
                    char_count=len(chunk_text),
                    section_heading=section_heading,
                    page_number=page_number,
                    strategy=self.strategy_name,
                    metadata={
                        **{k: v for k, v in meta.items()
                           if k not in ("source_file", "section_heading", "page_number")},
                    },
                )
            )

            idx += 1
            if end == len(text):
                break
            start += step

        logger.debug(
            "FixedSizeChunker produced %d chunks for document '%s' "
            "(chunk_size=%d, overlap=%d)",
            len(chunks),
            document_id,
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks
