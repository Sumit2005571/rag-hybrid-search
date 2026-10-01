"""Document metadata structures for ingestion pipeline."""

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Schema representing document-level metadata."""

    source_file: str = Field(..., description="Original file path or origin identifier")
    document_id: str = Field(..., description="Deterministic unique identifier for the document")
    file_type: str = Field(..., description="File extension (e.g. pdf, txt, md, html)")
    title: Optional[str] = Field(default=None, description="Document title if available")
    author: Optional[str] = Field(default=None, description="Author or creator")
    page_number: Optional[int] = Field(default=None, description="Page number (1-indexed) for page-based formats")
    total_pages: Optional[int] = Field(default=None, description="Total number of pages in the document")
    section_heading: Optional[str] = Field(default=None, description="Current section heading if detected")
    ingestion_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 timestamp of when the document was ingested",
    )
    extra: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary additional metadata")

    # Keep backward-compatible alias
    @property
    def source(self) -> str:
        return self.source_file


class RawDocument(BaseModel):
    """Container for loaded raw document content and its metadata."""

    content: str = Field(..., description="Extracted raw text content")
    metadata: DocumentMetadata = Field(..., description="Associated document metadata")


def generate_document_id(source_path: str, page_number: Optional[int] = None) -> str:
    """Generate a deterministic document ID from the source path and optional page number.

    This ensures re-ingesting the same file produces the same ID,
    enabling idempotent document indexing.

    Args:
        source_path: Absolute or relative path to the source file.
        page_number: Optional page number for multi-page documents.

    Returns:
        A stable hex digest string suitable for use as a document ID.
    """
    key = source_path
    if page_number is not None:
        key = f"{source_path}::page={page_number}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
