"""RAG Answer generator module."""

from typing import List, Optional
from pydantic import BaseModel, Field
from app.config import get_settings
from app.generation.citations import CitationExtractor
from app.generation.confidence import ConfidenceScorer, ConfidenceAssessment
from app.generation.prompts import build_rag_prompt
from app.retrieval.dense import RetrievalResult


class GenerationResult(BaseModel):
    """Result of RAG generation."""

    answer: str = Field(..., description="Generated answer text")
    citations: List[str] = Field(default_factory=list, description="Extracted citation identifiers")
    confidence: ConfidenceAssessment = Field(..., description="Confidence assessment")


class RAGGenerator:
    """Coordinates prompt construction, LLM generation, citation extraction, and confidence scoring."""

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.openai_chat_model
        self.citation_extractor = CitationExtractor()
        self.confidence_scorer = ConfidenceScorer(threshold=settings.confidence_threshold)

    def generate(self, question: str, retrieved_contexts: List[RetrievalResult]) -> GenerationResult:
        """Generate grounded answer using retrieved contexts.

        Note: LLM API call will be wired in generation milestone.
        """
        if not retrieved_contexts:
            assessment = ConfidenceAssessment(
                confidence_score=0.0,
                should_abstain=True,
                reason="No supporting contexts retrieved.",
            )
            return GenerationResult(
                answer="I don't know based on the provided documents.",
                citations=[],
                confidence=assessment,
            )

        # Foundation placeholder response
        answer = f"Based on retrieved documents, here is the answer to '{question}' [Chunk: {retrieved_contexts[0].chunk_id}]."
        citations = self.citation_extractor.extract(answer)
        scores = [ctx.score for ctx in retrieved_contexts]
        assessment = self.confidence_scorer.evaluate(answer, scores)

        return GenerationResult(
            answer=answer,
            citations=citations,
            confidence=assessment,
        )
