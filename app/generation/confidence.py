"""Answer confidence scoring and abstention handling."""

from typing import List
from pydantic import BaseModel, Field
from app.config import get_settings


class ConfidenceAssessment(BaseModel):
    """Assessment of generation confidence and abstention decision."""

    confidence_score: float = Field(..., description="Estimated confidence score (0.0 to 1.0)")
    should_abstain: bool = Field(..., description="Whether system should abstain from answering")
    reason: str = Field(default="", description="Explanation of confidence or abstention rationale")


class ConfidenceScorer:
    """Computes confidence score based on retrieval relevance and generation signals."""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def evaluate(self, answer: str, retrieval_scores: List[float]) -> ConfidenceAssessment:
        """Evaluate whether answer is confident or requires abstention."""
        # Simple heuristic stub for foundation milestone
        avg_retrieval_score = (sum(retrieval_scores) / len(retrieval_scores)) if retrieval_scores else 0.0

        is_unknown = "i don't know" in answer.lower() or "not enough information" in answer.lower()
        if is_unknown or avg_retrieval_score < self.threshold:
            return ConfidenceAssessment(
                confidence_score=avg_retrieval_score,
                should_abstain=True,
                reason="Context relevance is below confidence threshold or answer expressed uncertainty.",
            )

        return ConfidenceAssessment(
            confidence_score=avg_retrieval_score,
            should_abstain=False,
            reason="High confidence based on supporting retrieval evidence.",
        )
