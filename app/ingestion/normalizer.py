"""Document text normalization module."""

import re
from typing import Optional
from app.ingestion.metadata import RawDocument


class DocumentNormalizer:
    """Handles text cleaning, whitespace normalization, and encoding consistency."""

    def __init__(self, strip_extra_whitespace: bool = True, remove_control_chars: bool = True):
        self.strip_extra_whitespace = strip_extra_whitespace
        self.remove_control_chars = remove_control_chars

    def normalize_text(self, text: str) -> str:
        """Clean and normalize raw text."""
        if not text:
            return ""

        result = text

        if self.remove_control_chars:
            # Strip non-printable control characters except standard whitespace
            result = "".join(ch for ch in result if ch == "\n" or ch == "\t" or ch == "\r" or ch >= " ")

        if self.strip_extra_whitespace:
            # Normalize whitespace per line
            lines = [re.sub(r"[ \t]+", " ", line).strip() for line in result.splitlines()]
            result = "\n".join(lines)
            # Replace excessive newlines (3 or more) with double newlines
            result = re.sub(r"\n{3,}", "\n\n", result)
            result = result.strip()

        return result

    def normalize(self, document: RawDocument) -> RawDocument:
        """Normalize raw document content while preserving metadata."""
        cleaned_content = self.normalize_text(document.content)
        return RawDocument(content=cleaned_content, metadata=document.metadata)
