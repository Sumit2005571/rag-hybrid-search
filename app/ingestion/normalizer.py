"""Document text normalization module."""

import re
from app.ingestion.metadata import RawDocument
from app.logging_config import get_logger

logger = get_logger(__name__)


class DocumentNormalizer:
    """Handles text cleaning, whitespace normalization, and encoding consistency.

    All normalization is deterministic and produces the same output for the same input.
    """

    def __init__(self, strip_extra_whitespace: bool = True, remove_control_chars: bool = True):
        self.strip_extra_whitespace = strip_extra_whitespace
        self.remove_control_chars = remove_control_chars

    def normalize_text(self, text: str) -> str:
        """Clean and normalize raw text.

        Steps:
        1. Remove non-printable control characters (preserving ``\\n``, ``\\t``, ``\\r``).
        2. Collapse horizontal whitespace per line.
        3. Collapse excessive blank lines to a single blank line.
        4. Strip leading/trailing whitespace.

        Args:
            text: The raw text to normalize.

        Returns:
            Cleaned text string.
        """
        if not text:
            return ""

        result = text

        if self.remove_control_chars:
            # Strip non-printable control characters except standard whitespace
            result = "".join(
                ch for ch in result
                if ch == "\n" or ch == "\t" or ch == "\r" or ch >= " "
            )

        if self.strip_extra_whitespace:
            # Normalize whitespace per line
            lines = [re.sub(r"[ \t]+", " ", line).strip() for line in result.splitlines()]
            result = "\n".join(lines)
            # Replace excessive newlines (3 or more) with double newlines
            result = re.sub(r"\n{3,}", "\n\n", result)
            result = result.strip()

        return result

    def normalize(self, document: RawDocument) -> RawDocument:
        """Normalize raw document content while preserving metadata.

        Args:
            document: The raw document to normalize.

        Returns:
            A new ``RawDocument`` with normalized content and the same metadata.
        """
        cleaned_content = self.normalize_text(document.content)
        logger.debug(
            "Normalized document %s: %d chars -> %d chars",
            document.metadata.document_id,
            len(document.content),
            len(cleaned_content),
        )
        return RawDocument(content=cleaned_content, metadata=document.metadata)
