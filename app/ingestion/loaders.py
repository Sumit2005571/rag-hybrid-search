"""Multi-format document loaders."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union
from app.ingestion.metadata import RawDocument, DocumentMetadata


class BaseLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, source: Union[str, Path]) -> List[RawDocument]:
        """Load and parse document from given source path."""
        pass


class DocumentLoader(BaseLoader):
    """Factory loader dispatching to format-specific handlers (PDF, TXT, DOCX, Markdown)."""

    def __init__(self):
        pass

    def load(self, source: Union[str, Path]) -> List[RawDocument]:
        """Dispatch document loading based on file extension.

        Note: Detailed format-specific parsing will be implemented in the ingestion milestone.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        # Basic fallback reader for text-based files
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="latin-1", errors="replace")

        metadata = DocumentMetadata(
            source=str(path.resolve()),
            document_id=path.stem,
            title=path.name,
            file_type=path.suffix.lstrip(".").lower(),
        )
        return [RawDocument(content=content, metadata=metadata)]
