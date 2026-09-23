"""Document metadata structures."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Schema representing document-level metadata."""

    source: str = Field(..., description="File path, URL, or origin identifier")
    document_id: str = Field(..., description="Unique identifier for the document")
    title: Optional[str] = Field(default=None, description="Document title if available")
    file_type: Optional[str] = Field(default=None, description="MIME type or file extension (e.g. pdf, txt, docx)")
    author: Optional[str] = Field(default=None, description="Author or creator")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary additional metadata")


class RawDocument(BaseModel):
    """Container for loaded raw document content and its metadata."""

    content: str = Field(..., description="Extracted raw text content")
    metadata: DocumentMetadata = Field(..., description="Associated document metadata")
