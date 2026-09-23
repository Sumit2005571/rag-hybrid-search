"""Generation package supporting grounded answers, citations, verification, and confidence."""

from app.generation.prompts import build_rag_prompt
from app.generation.citations import Citation, CitationExtractor
from app.generation.verification import CitationVerifier, VerificationResult
from app.generation.confidence import ConfidenceScorer, ConfidenceAssessment
from app.generation.generator import RAGGenerator, GenerationResult

__all__ = [
    "build_rag_prompt",
    "Citation",
    "CitationExtractor",
    "CitationVerifier",
    "VerificationResult",
    "ConfidenceScorer",
    "ConfidenceAssessment",
    "RAGGenerator",
    "GenerationResult",
]
