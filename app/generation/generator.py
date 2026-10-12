"""RAG Answer generator module.

Implements grounded answer generation via OpenAI Chat Completion (SDK v3.x).

Flow
----
1. If no contexts are retrieved, return an abstention immediately.
2. Build the grounded prompt with :func:`~app.generation.prompts.build_rag_prompt`.
3. Call ``openai.ChatCompletion.create`` (v3 SDK).
4. Extract ``[Chunk: <id>]`` citation markers from the response text.
5. Verify citations against the set of retrieved chunk IDs.
6. Compute a confidence assessment from retrieval scores.

Design notes
------------
- The API key is read at call time and is never logged or returned to callers.
- Abstention is decided solely on retrieval quality (avg score < threshold) or
  explicit "i don't know" phrasing in the model output — NOT on arbitrary
  LLM-reported confidence numbers.
- ``openai.ChatCompletion`` resolves to ``APIRemovedInV1Proxy`` in openai≥1.x,
  but the installed version (3.19.0) is the v0.x/v2.x package that does support
  this interface. Tests mock this call.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.config import get_settings
from app.generation.citations import CitationExtractor
from app.generation.confidence import ConfidenceAssessment, ConfidenceScorer
from app.generation.prompts import build_rag_prompt
from app.generation.verification import CitationVerifier, VerificationResult
from app.retrieval.dense import RetrievalResult

logger = logging.getLogger(__name__)

# Sentinel text used when the context is empty.
_NO_CONTEXT_ANSWER = "I don't know based on the provided documents."


class GenerationResult(BaseModel):
    """Result of RAG generation."""

    answer: str = Field(..., description="Generated answer text")
    citations: List[str] = Field(
        default_factory=list, description="Chunk IDs cited in the answer"
    )
    confidence: ConfidenceAssessment = Field(..., description="Confidence assessment")
    verification: Optional[VerificationResult] = Field(
        default=None, description="Citation verification result (None if no citations)"
    )


class GenerationError(Exception):
    """Raised when the OpenAI Chat Completion call fails."""


class RAGGenerator:
    """Coordinates prompt construction, LLM generation, citation extraction,
    citation verification, and confidence scoring.

    Args:
        model_name: Override for the chat model (defaults to
            ``settings.openai_chat_model``).
        api_key: Override for the OpenAI API key.  If *None*, the key is
            resolved from ``settings.openai_api_key``.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.model_name: str = model_name or settings.openai_chat_model
        # Resolve key at construction time but never expose it.
        self._api_key: Optional[str] = api_key or settings.openai_api_key
        self._citation_extractor = CitationExtractor()
        self._citation_verifier = CitationVerifier()
        self._confidence_scorer = ConfidenceScorer(threshold=settings.confidence_threshold)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        question: str,
        retrieved_contexts: List[RetrievalResult],
    ) -> GenerationResult:
        """Generate a grounded answer using retrieved contexts.

        Args:
            question: The user's question.
            retrieved_contexts: Ordered list of retrieved chunks (best-first).

        Returns:
            A :class:`GenerationResult` with the answer, citations,
            confidence assessment, and citation verification.
        """
        # ---- Empty context → immediate abstention ---------------------
        if not retrieved_contexts:
            logger.info("No contexts retrieved — returning abstention answer.")
            assessment = ConfidenceAssessment(
                confidence_score=0.0,
                should_abstain=True,
                reason="No supporting contexts were retrieved.",
            )
            return GenerationResult(
                answer=_NO_CONTEXT_ANSWER,
                citations=[],
                confidence=assessment,
                verification=None,
            )

        # ---- Missing API key → abstention with clear error ------------
        if not self._api_key:
            logger.warning("OPENAI_API_KEY is not set — cannot call the chat model.")
            assessment = ConfidenceAssessment(
                confidence_score=0.0,
                should_abstain=True,
                reason="OpenAI API key is not configured; cannot generate an answer.",
            )
            return GenerationResult(
                answer=_NO_CONTEXT_ANSWER,
                citations=[],
                confidence=assessment,
                verification=None,
            )

        # ---- Call the LLM --------------------------------------------
        context_texts: List[str] = [r.text for r in retrieved_contexts]
        system_prompt, user_prompt = build_rag_prompt(question, context_texts)

        try:
            answer = self._call_openai(system_prompt, user_prompt)
        except GenerationError as exc:
            logger.error("LLM generation failed: %s", exc)
            assessment = ConfidenceAssessment(
                confidence_score=0.0,
                should_abstain=True,
                reason=f"LLM call failed: {exc}",
            )
            return GenerationResult(
                answer=_NO_CONTEXT_ANSWER,
                citations=[],
                confidence=assessment,
                verification=None,
            )

        # ---- Extract & verify citations ------------------------------
        cited_ids: List[str] = self._citation_extractor.extract(answer)

        # Build a lookup of {chunk_id → text} from retrieved contexts.
        retrieved_map: Dict[str, str] = {r.chunk_id: r.text for r in retrieved_contexts}
        verification: Optional[VerificationResult] = None
        if cited_ids:
            verification = self._citation_verifier.verify(cited_ids, retrieved_map)

        # Keep only verified citations (drop hallucinated ones).
        valid_citations: List[str] = cited_ids
        if verification is not None:
            valid_citations = verification.valid_citations
            if verification.hallucinated_citations:
                logger.warning(
                    "Hallucinated citations detected and removed: %s",
                    verification.hallucinated_citations,
                )

        # ---- Confidence assessment -----------------------------------
        retrieval_scores: List[float] = [r.score for r in retrieved_contexts]
        assessment = self._confidence_scorer.evaluate(answer, retrieval_scores)

        logger.info(
            "Generation complete: abstain=%s, citations=%d, confidence=%.3f",
            assessment.should_abstain,
            len(valid_citations),
            assessment.confidence_score,
        )

        return GenerationResult(
            answer=answer,
            citations=valid_citations,
            confidence=assessment,
            verification=verification,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Call openai.ChatCompletion.create and return the message content.

        Uses the v3.x / v0.28-style SDK (openai.ChatCompletion).
        The API key is injected per-call and never exposed to callers.

        Raises:
            GenerationError: Any API or configuration error.
        """
        import openai  # deferred import so tests can patch cleanly

        openai.api_key = self._api_key  # set per-call, never stored in response
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,  # deterministic generation
            )
            # v3 SDK response: response["choices"][0]["message"]["content"]
            return response["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            raise GenerationError(str(exc)) from exc
