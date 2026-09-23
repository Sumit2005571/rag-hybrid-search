"""Fixed-size chunking implementation."""

from typing import Any, Dict, List, Optional
from app.chunking.base import BaseChunker, Chunk


class FixedSizeChunker(BaseChunker):
    """Chunks text into fixed-size windows with configurable overlap."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, document_id: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text into fixed-size character windows."""
        if not text:
            return []

        chunks: List[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        idx = 0
        base_meta = metadata or {}

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}#c{idx}",
                    document_id=document_id,
                    text=chunk_text,
                    chunk_index=idx,
                    metadata={**base_meta, "start_char": start, "end_char": end},
                )
            )
            idx += 1
            if end == len(text):
                break
            start += step

        return chunks
