"""Recursive structure-aware chunking interface."""

from typing import Any, Dict, List, Optional
from app.chunking.base import BaseChunker, Chunk


class RecursiveStructureChunker(BaseChunker):
    """Splits text recursively using structural separators (paragraphs, sentences, words)."""

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(self, text: str, document_id: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Placeholder for recursive structure-aware splitting logic."""
        # Full implementation will be added in chunking milestone
        if not text:
            return []
        base_meta = metadata or {}
        return [
            Chunk(
                chunk_id=f"{document_id}#rec0",
                document_id=document_id,
                text=text[:self.chunk_size],
                chunk_index=0,
                metadata=base_meta,
            )
        ]
