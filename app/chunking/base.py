"""Base interfaces and data structures for document chunking."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Container for a single chunk of text and its metadata."""

    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    document_id: str = Field(..., description="Parent document identifier")
    text: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(..., description="0-indexed position within document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata tags and attributes")


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, document_id: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text into a list of Chunk objects."""
        pass
