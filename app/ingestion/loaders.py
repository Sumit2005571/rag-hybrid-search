"""Multi-format document loaders with pluggable format support.

Supported formats:
  - Plain text (.txt)
  - Markdown (.md)
  - HTML (.html, .htm)
  - PDF (.pdf)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

from app.ingestion.metadata import DocumentMetadata, RawDocument, generate_document_id
from app.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseLoader(ABC):
    """Abstract base class for document loaders.

    Implement ``load`` for each file format. Register subclasses using
    :func:`DocumentLoader.register` so the dispatcher can route by extension.
    """

    @abstractmethod
    def load(self, source: Union[str, Path]) -> List[RawDocument]:
        """Load and parse document(s) from the given source path.

        Args:
            source: Filesystem path to the document.

        Returns:
            List of ``RawDocument`` objects (one per logical page/section).
        """
        ...


# ---------------------------------------------------------------------------
# TXT loader
# ---------------------------------------------------------------------------

class TextLoader(BaseLoader):
    """Loads plain-text files while preserving useful structure."""

    def load(self, source: Union[str, Path]) -> List[RawDocument]:
        path = Path(source)
        logger.info("Loading TXT: %s", path.name)

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("UTF-8 decode failed for %s; falling back to latin-1", path.name)
            content = path.read_text(encoding="latin-1", errors="replace")

        doc_id = generate_document_id(str(path.resolve()))
        metadata = DocumentMetadata(
            source_file=str(path.resolve()),
            document_id=doc_id,
            file_type="txt",
            title=path.name,
        )
        return [RawDocument(content=content, metadata=metadata)]


# ---------------------------------------------------------------------------
# Markdown loader
# ---------------------------------------------------------------------------

class MarkdownLoader(BaseLoader):
    """Loads Markdown files, detecting heading hierarchy and producing
    per-section ``RawDocument`` objects when headings are present."""

    def load(self, source: Union[str, Path]) -> List[RawDocument]:
        path = Path(source)
        logger.info("Loading Markdown: %s", path.name)

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="latin-1", errors="replace")

        sections = self._split_by_headings(content)
        base_source = str(path.resolve())

        documents: List[RawDocument] = []
        if len(sections) <= 1:
            # No heading structure detected — return the whole document
            doc_id = generate_document_id(base_source)
            metadata = DocumentMetadata(
                source_file=base_source,
                document_id=doc_id,
                file_type="md",
                title=path.stem,
            )
            documents.append(RawDocument(content=content, metadata=metadata))
        else:
            for idx, (heading, body) in enumerate(sections):
                doc_id = generate_document_id(base_source, page_number=idx)
                metadata = DocumentMetadata(
                    source_file=base_source,
                    document_id=doc_id,
                    file_type="md",
                    title=path.stem,
                    section_heading=heading,
                    extra={"section_index": idx},
                )
                full_text = f"{heading}\n\n{body}".strip() if heading else body.strip()
                documents.append(RawDocument(content=full_text, metadata=metadata))

        return documents

    @staticmethod
    def _split_by_headings(text: str) -> List[tuple]:
        """Split markdown text into ``(heading, body)`` tuples.

        Returns a list of ``(heading_line | None, body_text)`` pairs.
        A preamble before any heading yields ``(None, preamble_text)``.
        """
        import re

        heading_pattern = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)
        positions = [(m.start(), m.group(1)) for m in heading_pattern.finditer(text)]

        if not positions:
            return [(None, text)]

        sections: List[tuple] = []

        # Preamble before first heading
        if positions[0][0] > 0:
            preamble = text[: positions[0][0]].strip()
            if preamble:
                sections.append((None, preamble))

        for i, (pos, heading) in enumerate(positions):
            end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
            body = text[pos + len(heading) : end].strip()
            sections.append((heading, body))

        return sections


# ---------------------------------------------------------------------------
# HTML loader
# ---------------------------------------------------------------------------

class HTMLLoader(BaseLoader):
    """Loads HTML files, stripping noise (scripts, styles, nav) and
    preserving meaningful heading text."""

    # Tags to remove entirely
    NOISE_TAGS = {"script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"}

    def load(self, source: Union[str, Path]) -> List[RawDocument]:
        path = Path(source)
        logger.info("Loading HTML: %s", path.name)

        try:
            raw_html = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw_html = path.read_text(encoding="latin-1", errors="replace")

        cleaned_text, title = self._extract_text(raw_html)
        doc_id = generate_document_id(str(path.resolve()))
        metadata = DocumentMetadata(
            source_file=str(path.resolve()),
            document_id=doc_id,
            file_type="html",
            title=title or path.stem,
        )
        return [RawDocument(content=cleaned_text, metadata=metadata)]

    def _extract_text(self, html: str) -> tuple:
        """Strip noise, return ``(cleaned_text, title)``."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")

        # Extract title before we remove anything
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else None

        # Remove noise elements
        for tag_name in self.NOISE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Mark headings with line-prefix markers so they survive get_text()
        for level in range(1, 7):
            for heading in soup.find_all(f"h{level}"):
                prefix = "#" * level
                heading.string = f"\n{prefix} {heading.get_text(strip=True)}\n"

        text = soup.get_text(separator="\n")
        return text, title


# ---------------------------------------------------------------------------
# PDF loader
# ---------------------------------------------------------------------------

class PDFLoader(BaseLoader):
    """Loads PDF files page-by-page using PyMuPDF (fitz), preserving
    per-page content and page numbers."""

    def load(self, source: Union[str, Path]) -> List[RawDocument]:
        path = Path(source)
        logger.info("Loading PDF: %s", path.name)

        try:
            import pymupdf as fitz
        except ImportError:
            import fitz

        documents: List[RawDocument] = []
        base_source = str(path.resolve())

        try:
            doc = fitz.open(str(path))
        except Exception as exc:
            logger.error("Failed to open PDF %s: %s", path.name, exc)
            raise

        total_pages = len(doc)
        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text("text")
            doc_id = generate_document_id(base_source, page_number=page_num + 1)
            metadata = DocumentMetadata(
                source_file=base_source,
                document_id=doc_id,
                file_type="pdf",
                title=path.stem,
                page_number=page_num + 1,
                total_pages=total_pages,
            )
            documents.append(RawDocument(content=text, metadata=metadata))

        doc.close()
        logger.info("Extracted %d pages from %s", total_pages, path.name)
        return documents


# ---------------------------------------------------------------------------
# Dispatcher / registry
# ---------------------------------------------------------------------------

class DocumentLoader:
    """Factory loader dispatching to format-specific handlers by file extension.

    New formats can be added via :meth:`register`:

    .. code-block:: python

        loader = DocumentLoader()
        loader.register(".csv", CSVLoader())
        docs = loader.load("data.csv")
    """

    # Default extension -> loader mapping
    _DEFAULT_REGISTRY: Dict[str, BaseLoader] = {
        ".txt": TextLoader(),
        ".md": MarkdownLoader(),
        ".html": HTMLLoader(),
        ".htm": HTMLLoader(),
        ".pdf": PDFLoader(),
    }

    def __init__(self):
        self._registry: Dict[str, BaseLoader] = dict(self._DEFAULT_REGISTRY)

    # ---- public API --------------------------------------------------------

    def register(self, extension: str, loader: BaseLoader) -> None:
        """Register a loader for a given file extension.

        Args:
            extension: File extension including the dot (e.g. ``'.csv'``).
            loader: An instance of a ``BaseLoader`` subclass.
        """
        ext = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        self._registry[ext] = loader
        logger.info("Registered loader for extension %s", ext)

    def load(self, source: Union[str, Path]) -> List[RawDocument]:
        """Dispatch document loading based on file extension.

        Args:
            source: Path to the file to ingest.

        Returns:
            List of ``RawDocument`` objects.

        Raises:
            FileNotFoundError: If source does not exist.
            ValueError: If no loader is registered for the file extension.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        ext = path.suffix.lower()
        loader = self._registry.get(ext)
        if loader is None:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Registered extensions: {list(self._registry.keys())}"
            )

        logger.info("Dispatching %s to %s", path.name, type(loader).__name__)
        return loader.load(path)

    @property
    def supported_extensions(self) -> List[str]:
        """Return list of currently supported file extensions."""
        return list(self._registry.keys())
