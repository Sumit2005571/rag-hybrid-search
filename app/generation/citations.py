"""Citation extraction and formatting."""

import re
from typing import List
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Citation reference supporting an answer claim."""

    chunk_id: str = Field(..., description="Referenced chunk identifier")
    source: str = Field(default="", description="Source file or document")
    quote: str = Field(default="", description="Quoted or excerpted context snippet")


class CitationExtractor:
    """Extracts citation markers from generated answer text."""

    CITATION_PATTERN = re.compile(r"\[Chunk:\s*([^\]]+)\]")

    def extract(self, text: str) -> List[str]:
        """Extract all chunk IDs cited in text."""
        return self.CITATION_PATTERN.findall(text)
