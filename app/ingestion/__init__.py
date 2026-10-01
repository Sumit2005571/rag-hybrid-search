"""Document ingestion and preprocessing package."""

from app.ingestion.metadata import DocumentMetadata, RawDocument, generate_document_id
from app.ingestion.normalizer import DocumentNormalizer
from app.ingestion.loaders import (
    BaseLoader,
    DocumentLoader,
    TextLoader,
    MarkdownLoader,
    HTMLLoader,
    PDFLoader,
)

__all__ = [
    "DocumentMetadata",
    "RawDocument",
    "generate_document_id",
    "DocumentNormalizer",
    "BaseLoader",
    "DocumentLoader",
    "TextLoader",
    "MarkdownLoader",
    "HTMLLoader",
    "PDFLoader",
]
