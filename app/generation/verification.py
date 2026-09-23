"""Citation and factual consistency verification."""

from typing import Dict, List
from pydantic import BaseModel, Field


class VerificationResult(BaseModel):
    """Result of citation verification."""

    is_verified: bool = Field(..., description="Whether citations are grounded in context")
    valid_citations: List[str] = Field(default_factory=list, description="Citations verified to exist")
    hallucinated_citations: List[str] = Field(default_factory=list, description="Citations not found in context")
    verification_score: float = Field(default=1.0, description="Verification reliability score between 0 and 1")


class CitationVerifier:
    """Verifies that generated citations match retrieved context chunks."""

    def verify(self, cited_chunk_ids: List[str], available_chunks: Dict[str, str]) -> VerificationResult:
        """Check if cited chunk IDs exist in the retrieved context."""
        valid: List[str] = []
        hallucinated: List[str] = []

        for cid in cited_chunk_ids:
            if cid in available_chunks:
                valid.append(cid)
            else:
                hallucinated.append(cid)

        total = len(cited_chunk_ids)
        score = (len(valid) / total) if total > 0 else 1.0

        return VerificationResult(
            is_verified=len(hallucinated) == 0,
            valid_citations=valid,
            hallucinated_citations=hallucinated,
            verification_score=score,
        )
