"""Recursive structure-aware chunking.

Splits a document by trying a priority-ordered list of separators.  The
algorithm prefers larger structural boundaries (section headings, blank lines
between paragraphs) over smaller ones (sentences, then words, then individual
characters) and only falls back to smaller boundaries when a segment would
exceed *chunk_size*.

Separator priority (from most to least preferred):
    1. ``\\n## `` / ``\\n# `` — Markdown heading boundaries
    2. ``\\n\\n`` — Paragraph breaks
    3. ``\\n`` — Line breaks
    4. ``". "`` / ``"? "`` / ``"! "`` — Sentence endings
    5. ``" "`` — Word boundaries
    6. ``""`` — Hard character split (last resort)

Each resulting segment that still exceeds *chunk_size* is recursively split
with the next separator in the list until it fits.  An optional *overlap*
copies the tail of the previous chunk to the beginning of the next one.
"""

import re
from typing import Any, Dict, List, Optional

from app.chunking.base import BaseChunker, Chunk, generate_chunk_id
from app.logging_config import get_logger

logger = get_logger(__name__)

# Ordered separators: try heading breaks first, fall back to characters.
_DEFAULT_SEPARATORS: List[str] = [
    "\n## ",   # Markdown h2
    "\n# ",    # Markdown h1
    "\n### ",  # Markdown h3
    "\n#### ", # Markdown h4
    "\n\n",    # Paragraph break
    "\n",      # Line break
    ". ",      # Sentence end
    "? ",      # Question end
    "! ",      # Exclamation end
    " ",       # Word boundary
    "",        # Hard character split
]


class RecursiveStructureChunker(BaseChunker):
    """Split text recursively using structural separators.

    The chunker tries the highest-priority separator first.  Any resulting
    segment that still exceeds *chunk_size* is recursively split with the next
    separator in the list.

    Args:
        chunk_size: Target maximum number of characters per chunk.
        chunk_overlap: Characters from the end of one chunk prepended to the
            next.  Must be strictly less than *chunk_size*.
        separators: Override the default separator list.  The list is tried in
            order; earlier entries are preferred.

    Example::

        chunker = RecursiveStructureChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk(text=doc.content, document_id=doc.metadata.document_id)
    """

    strategy_name: str = "recursive"

    DEFAULT_SEPARATORS: List[str] = _DEFAULT_SEPARATORS

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than "
                f"chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators: List[str] = separators if separators is not None else list(self.DEFAULT_SEPARATORS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        text: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Split *text* recursively at structural boundaries.

        Args:
            text: Document text to split.
            document_id: Deterministic parent document ID.
            metadata: Document-level metadata propagated to each chunk.

        Returns:
            Ordered list of :class:`~app.chunking.base.Chunk` objects.
        """
        if not text:
            return []

        meta = metadata or {}
        source_file = meta.get("source_file", "")
        section_heading = meta.get("section_heading")
        page_number = meta.get("page_number")

        # Gather raw text segments via recursive splitting
        raw_segments = self._split_recursive(text, self.separators, 0)

        # Apply overlap and build Chunk objects
        chunks: List[Chunk] = []
        accumulated_offset = 0  # approximate char offset tracking
        overlap_tail = ""

        for idx, segment in enumerate(raw_segments):
            chunk_text = (overlap_tail + segment) if overlap_tail else segment
            start = max(0, accumulated_offset - len(overlap_tail))

            chunks.append(
                Chunk(
                    chunk_id=generate_chunk_id(document_id, idx, self.strategy_name),
                    document_id=document_id,
                    source_file=source_file,
                    chunk_index=idx,
                    start_char=start,
                    end_char=start + len(chunk_text),
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

            accumulated_offset += len(segment)
            # Compute the overlap tail for the next chunk
            if self.chunk_overlap > 0 and len(chunk_text) > self.chunk_overlap:
                overlap_tail = chunk_text[-self.chunk_overlap:]
            else:
                overlap_tail = ""

        logger.debug(
            "RecursiveStructureChunker produced %d chunks for document '%s'",
            len(chunks),
            document_id,
        )
        return chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_recursive(
        self,
        text: str,
        separators: List[str],
        depth: int,
    ) -> List[str]:
        """Recursively split *text* using the separator priority list.

        If the text fits within *chunk_size* it is returned as-is.  Otherwise
        it is split on the first separator in *separators*; segments that are
        still too large are recursively split with the remaining separators.

        Args:
            text: Text to split.
            separators: Remaining separators to try (in priority order).
            depth: Current recursion depth (for guard against infinite loops).

        Returns:
            List of text segments, each within *chunk_size* characters
            (best-effort; hard-character fallback guarantees it).
        """
        if not text:
            return []

        # Base case: text fits in one chunk
        if len(text) <= self.chunk_size:
            return [text]

        # No more separators left — hard character split
        if not separators:
            return self._hard_split(text)

        sep = separators[0]
        remaining = separators[1:]

        if sep == "":
            return self._hard_split(text)

        # Split on the separator (keep the separator at the start of each part
        # after the first so we don't lose structural markers).
        parts = self._split_on_separator(text, sep)

        if len(parts) == 1:
            # Separator not found; try next
            return self._split_recursive(text, remaining, depth + 1)

        # Merge short consecutive parts to avoid producing many tiny chunks
        merged = self._merge_small_parts(parts, sep)

        result: List[str] = []
        for part in merged:
            if len(part) <= self.chunk_size:
                result.append(part)
            else:
                # Recurse with remaining separators
                result.extend(self._split_recursive(part, remaining, depth + 1))

        return result

    @staticmethod
    def _split_on_separator(text: str, sep: str) -> List[str]:
        """Split *text* on *sep*, prepending the separator to all parts
        after the first so heading markers are preserved."""
        parts = text.split(sep)
        if len(parts) == 1:
            return parts
        # Re-attach the separator to the start of each subsequent part
        result = [parts[0]]
        for p in parts[1:]:
            result.append(sep + p)
        return result

    def _merge_small_parts(self, parts: List[str], sep: str) -> List[str]:
        """Merge consecutive parts that are individually smaller than chunk_size
        until merging would exceed chunk_size."""
        merged: List[str] = []
        current = ""
        for part in parts:
            candidate = current + part if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                current = part
        if current:
            merged.append(current)
        return merged

    def _hard_split(self, text: str) -> List[str]:
        """Split *text* into chunks of exactly *chunk_size* characters."""
        return [
            text[i: i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size)
        ]
