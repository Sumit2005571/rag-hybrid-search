"""Comprehensive mocked unit tests for the end-to-end /query pipeline.

Coverage
--------
- Full /query flow: retrieval → generation → citation verification → response
- Empty retrieval context → abstention response
- Missing OpenAI API key → 503 from route; abstention from generator
- Retrieval failure → 503 from route
- Malformed request (empty query) → 422
- Grounded generation with citations
- Hallucinated citation removal
- Confidence below threshold → abstained=True
- OpenAI API error → abstention fallback
- Citation extraction and verification (unit)
- Confidence scorer (unit)

All tests use mocks — no network access, no OpenAI credits, no model downloads.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.generation.citations import CitationExtractor
from app.generation.confidence import ConfidenceAssessment, ConfidenceScorer
from app.generation.generator import GenerationError, RAGGenerator
from app.generation.verification import CitationVerifier
from app.main import app
from app.retrieval.dense import RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    chunk_id: str = "doc1#c0",
    text: str = "The sky is blue.",
    score: float = 0.9,
    document_id: str = "doc1",
    source_file: str = "sample.txt",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        score=score,
        rank=1,
        metadata={"source_file": source_file},
    )


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _patch_retriever(results: List[RetrievalResult] = None):
    """Return a mock retriever that returns *results* from .retrieve()."""
    mock = MagicMock()
    mock.retrieve.return_value = results or []
    return mock


# ---------------------------------------------------------------------------
# /query endpoint — integration-level (mocked dependencies)
# ---------------------------------------------------------------------------


class TestQueryEndpoint:
    """Tests for POST /api/v1/query with mocked retrieval and generation."""

    def test_full_flow_grounded_answer(self, client: TestClient):
        """Full pipeline: retrieval returns chunks → LLM returns grounded answer."""
        chunk = _make_result("doc1#c0", "Hybrid search combines dense and sparse retrieval.", 0.95)
        mock_retriever = _patch_retriever([chunk])

        mock_gen = MagicMock()
        mock_gen.answer = "Hybrid search combines dense and sparse retrieval. [Chunk: doc1#c0]"
        mock_gen.citations = ["doc1#c0"]
        mock_gen.confidence = ConfidenceAssessment(
            confidence_score=0.95, should_abstain=False, reason="High confidence"
        )
        from app.generation.verification import VerificationResult
        mock_gen.verification = VerificationResult(
            is_verified=True, valid_citations=["doc1#c0"], hallucinated_citations=[], verification_score=1.0
        )

        with patch("app.api.routes._build_hybrid_retriever", return_value=mock_retriever), \
             patch("app.api.routes.RAGGenerator") as MockRAG:
            MockRAG.return_value.generate.return_value = mock_gen
            resp = client.post("/api/v1/query", json={"query": "What is hybrid search?", "top_k": 5})

        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "What is hybrid search?"
        assert "hybrid search" in data["answer"].lower()
        assert data["abstained"] is False
        assert data["confidence_score"] == pytest.approx(0.95)
        assert len(data["citations"]) == 1
        assert data["citations"][0]["chunk_id"] == "doc1#c0"
        assert data["citations"][0]["source"] == "sample.txt"
        assert data["verification_score"] == pytest.approx(1.0)
        assert len(data["retrieved_chunks"]) == 1

    def test_empty_retrieval_returns_abstention(self, client: TestClient):
        """When no chunks are retrieved, response must be abstained."""
        mock_retriever = _patch_retriever([])

        mock_gen = MagicMock()
        mock_gen.answer = "I don't know based on the provided documents."
        mock_gen.citations = []
        mock_gen.confidence = ConfidenceAssessment(
            confidence_score=0.0, should_abstain=True, reason="No contexts retrieved."
        )
        mock_gen.verification = None

        with patch("app.api.routes._build_hybrid_retriever", return_value=mock_retriever), \
             patch("app.api.routes.RAGGenerator") as MockRAG:
            MockRAG.return_value.generate.return_value = mock_gen
            resp = client.post("/api/v1/query", json={"query": "Unknown topic"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["abstained"] is True
        assert "don't know" in data["answer"].lower()
        assert data["citations"] == []
        assert data["retrieved_chunks"] == []
        assert data["verification_score"] is None

    def test_missing_api_key_returns_503(self, client: TestClient):
        """/query returns 503 when the retriever cannot be built (missing API key)."""
        with patch(
            "app.api.routes._build_hybrid_retriever",
            side_effect=Exception("OPENAI_API_KEY is not set"),
        ):
            resp = client.post("/api/v1/query", json={"query": "Some question"})

        assert resp.status_code == 503
        detail = resp.json()["detail"]
        assert "unavailable" in detail.lower() or "OPENAI_API_KEY" in detail or "index" in detail.lower()

    def test_retrieval_failure_returns_503(self, client: TestClient):
        """/query returns 503 when the hybrid retriever itself raises."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = RuntimeError("ChromaDB connection failed")

        with patch("app.api.routes._build_hybrid_retriever", return_value=mock_retriever):
            resp = client.post("/api/v1/query", json={"query": "Database question"})

        assert resp.status_code == 503

    def test_empty_query_returns_422(self, client: TestClient):
        """Pydantic validation rejects an empty query string."""
        resp = client.post("/api/v1/query", json={"query": ""})
        assert resp.status_code == 422

    def test_top_k_forwarded_to_retriever(self, client: TestClient):
        """The top_k request parameter is passed to the retriever."""
        mock_retriever = _patch_retriever([])
        mock_gen = MagicMock()
        mock_gen.answer = "I don't know based on the provided documents."
        mock_gen.citations = []
        mock_gen.confidence = ConfidenceAssessment(
            confidence_score=0.0, should_abstain=True, reason="No contexts."
        )
        mock_gen.verification = None

        with patch("app.api.routes._build_hybrid_retriever", return_value=mock_retriever), \
             patch("app.api.routes.RAGGenerator") as MockRAG:
            MockRAG.return_value.generate.return_value = mock_gen
            client.post("/api/v1/query", json={"query": "test", "top_k": 7})

        mock_retriever.retrieve.assert_called_once()
        call_kwargs = mock_retriever.retrieve.call_args
        assert call_kwargs.kwargs.get("top_k") == 7 or call_kwargs.args[1] == 7

    def test_use_reranker_false_forwarded(self, client: TestClient):
        """use_reranker=False is forwarded to the retriever."""
        mock_retriever = _patch_retriever([])
        mock_gen = MagicMock()
        mock_gen.answer = "I don't know based on the provided documents."
        mock_gen.citations = []
        mock_gen.confidence = ConfidenceAssessment(
            confidence_score=0.0, should_abstain=True, reason="No contexts."
        )
        mock_gen.verification = None

        with patch("app.api.routes._build_hybrid_retriever", return_value=mock_retriever), \
             patch("app.api.routes.RAGGenerator") as MockRAG:
            MockRAG.return_value.generate.return_value = mock_gen
            client.post("/api/v1/query", json={"query": "test", "use_reranker": False})

        call_kwargs = mock_retriever.retrieve.call_args
        # use_reranker may be positional or keyword
        all_args = list(call_kwargs.args) + list(call_kwargs.kwargs.values())
        assert False in all_args or call_kwargs.kwargs.get("use_reranker") is False

    def test_multiple_citations_in_response(self, client: TestClient):
        """Multiple verified citations are returned as CitationItem objects."""
        chunks = [
            _make_result("doc1#c0", "BM25 is a sparse retrieval method.", 0.85),
            _make_result("doc1#c1", "Dense retrieval uses embeddings.", 0.80),
        ]
        mock_retriever = _patch_retriever(chunks)

        from app.generation.verification import VerificationResult
        mock_gen = MagicMock()
        mock_gen.answer = (
            "BM25 is sparse. [Chunk: doc1#c0] Dense retrieval uses vectors. [Chunk: doc1#c1]"
        )
        mock_gen.citations = ["doc1#c0", "doc1#c1"]
        mock_gen.confidence = ConfidenceAssessment(
            confidence_score=0.83, should_abstain=False, reason="Good evidence"
        )
        mock_gen.verification = VerificationResult(
            is_verified=True,
            valid_citations=["doc1#c0", "doc1#c1"],
            hallucinated_citations=[],
            verification_score=1.0,
        )

        with patch("app.api.routes._build_hybrid_retriever", return_value=mock_retriever), \
             patch("app.api.routes.RAGGenerator") as MockRAG:
            MockRAG.return_value.generate.return_value = mock_gen
            resp = client.post("/api/v1/query", json={"query": "Compare BM25 and dense retrieval"})

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["citations"]) == 2
        chunk_ids = {c["chunk_id"] for c in data["citations"]}
        assert "doc1#c0" in chunk_ids
        assert "doc1#c1" in chunk_ids

    def test_api_key_never_in_response(self, client: TestClient):
        """The OpenAI API key must never appear in the response body."""
        mock_retriever = _patch_retriever([])
        mock_gen = MagicMock()
        mock_gen.answer = "I don't know based on the provided documents."
        mock_gen.citations = []
        mock_gen.confidence = ConfidenceAssessment(
            confidence_score=0.0, should_abstain=True, reason="No contexts."
        )
        mock_gen.verification = None

        fake_key = "sk-fakekeythatmustnotappear"
        with patch("app.api.routes._build_hybrid_retriever", return_value=mock_retriever), \
             patch("app.api.routes.RAGGenerator") as MockRAG, \
             patch.dict("os.environ", {"OPENAI_API_KEY": fake_key}):
            MockRAG.return_value.generate.return_value = mock_gen
            resp = client.post("/api/v1/query", json={"query": "Sensitive question"})

        assert fake_key not in resp.text


# ---------------------------------------------------------------------------
# RAGGenerator unit tests
# ---------------------------------------------------------------------------


class TestRAGGeneratorUnit:
    """Unit tests for RAGGenerator with mocked OpenAI calls."""

    @pytest.fixture(autouse=True)
    def _set_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    def _make_generator(self) -> RAGGenerator:
        return RAGGenerator(model_name="gpt-4o-mini", api_key="sk-test-key")

    def test_empty_contexts_returns_abstention(self):
        """Empty retrieved_contexts → abstained answer without calling LLM."""
        gen = self._make_generator()
        result = gen.generate("What is RAG?", [])
        assert result.confidence.should_abstain is True
        assert "don't know" in result.answer.lower()
        assert result.citations == []
        assert result.verification is None

    def test_missing_api_key_returns_abstention(self):
        """No API key → abstention without calling LLM."""
        gen = RAGGenerator(model_name="gpt-4o-mini", api_key=None)
        # Ensure settings don't supply a key either
        with patch("app.generation.generator.get_settings") as mock_settings:
            mock_settings.return_value.openai_api_key = None
            mock_settings.return_value.openai_chat_model = "gpt-4o-mini"
            mock_settings.return_value.confidence_threshold = 0.7
            gen2 = RAGGenerator(model_name="gpt-4o-mini", api_key=None)
            result = gen2.generate("What is RAG?", [_make_result()])
        assert result.confidence.should_abstain is True

    def test_grounded_answer_with_valid_citation(self):
        """LLM returns answer with valid citation → citation in result."""
        ctx = _make_result("doc1#c0", "Hybrid search is effective.", 0.9)
        gen = self._make_generator()

        mock_response = {
            "choices": [{"message": {"content": "It is effective. [Chunk: doc1#c0]"}}]
        }
        with patch("openai.ChatCompletion.create", return_value=mock_response):
            result = gen.generate("Is hybrid search effective?", [ctx])

        assert "doc1#c0" in result.citations
        assert result.verification is not None
        assert result.verification.is_verified is True
        assert result.verification.hallucinated_citations == []

    def test_hallucinated_citation_removed(self):
        """Citation to a chunk not in retrieved results is removed from output."""
        ctx = _make_result("doc1#c0", "Dense retrieval uses vectors.", 0.85)
        gen = self._make_generator()

        # LLM cites a chunk that was NOT retrieved
        mock_response = {
            "choices": [{"message": {"content": "Something [Chunk: doc1#c0] fake [Chunk: ghost#c99]"}}]
        }
        with patch("openai.ChatCompletion.create", return_value=mock_response):
            result = gen.generate("Tell me about retrieval.", [ctx])

        assert "doc1#c0" in result.citations
        assert "ghost#c99" not in result.citations
        assert result.verification is not None
        assert "ghost#c99" in result.verification.hallucinated_citations
        assert result.verification.verification_score < 1.0

    def test_openai_api_error_returns_abstention(self):
        """OpenAI API error → GenerationError caught → abstention result returned."""
        ctx = _make_result()
        gen = self._make_generator()

        with patch("openai.ChatCompletion.create", side_effect=Exception("APIConnectionError")):
            result = gen.generate("What is retrieval?", [ctx])

        assert result.confidence.should_abstain is True
        assert "don't know" in result.answer.lower()
        assert result.citations == []

    def test_confidence_below_threshold_abstains(self):
        """Low retrieval scores → confidence below threshold → should_abstain=True."""
        # Score of 0.1 is below the default threshold of 0.7
        ctx = _make_result(score=0.1)
        gen = self._make_generator()

        mock_response = {
            "choices": [{"message": {"content": "Some answer with no citations."}}]
        }
        with patch("openai.ChatCompletion.create", return_value=mock_response):
            result = gen.generate("Low confidence question?", [ctx])

        assert result.confidence.should_abstain is True
        assert result.confidence.confidence_score == pytest.approx(0.1)

    def test_high_confidence_does_not_abstain(self):
        """High retrieval scores + answer without uncertainty → should_abstain=False."""
        ctx = _make_result(score=0.95)
        gen = self._make_generator()

        mock_response = {
            "choices": [{"message": {"content": "The answer is correct. [Chunk: doc1#c0]"}}]
        }
        with patch("openai.ChatCompletion.create", return_value=mock_response):
            result = gen.generate("High confidence question?", [ctx])

        assert result.confidence.should_abstain is False
        assert result.confidence.confidence_score == pytest.approx(0.95)

    def test_api_key_not_exposed_in_result(self):
        """The OpenAI API key must not appear anywhere in the GenerationResult."""
        fake_key = "sk-superstrictly-private-key"
        gen = RAGGenerator(model_name="gpt-4o-mini", api_key=fake_key)
        ctx = _make_result()

        mock_response = {
            "choices": [{"message": {"content": "Normal answer."}}]
        }
        with patch("openai.ChatCompletion.create", return_value=mock_response):
            result = gen.generate("Question?", [ctx])

        result_str = result.model_dump_json()
        assert fake_key not in result_str

    def test_no_citations_in_answer_gives_empty_list(self):
        """Answer with no [Chunk: ...] markers → empty citation list."""
        ctx = _make_result()
        gen = self._make_generator()

        mock_response = {
            "choices": [{"message": {"content": "This is an answer without any citations."}}]
        }
        with patch("openai.ChatCompletion.create", return_value=mock_response):
            result = gen.generate("Simple question?", [ctx])

        assert result.citations == []
        assert result.verification is None  # no citations → no verification needed


# ---------------------------------------------------------------------------
# CitationExtractor unit tests
# ---------------------------------------------------------------------------


class TestCitationExtractor:
    """Unit tests for citation pattern extraction."""

    def test_extracts_single_citation(self):
        extractor = CitationExtractor()
        text = "The answer is yes. [Chunk: doc1#c0]"
        assert extractor.extract(text) == ["doc1#c0"]

    def test_extracts_multiple_citations(self):
        extractor = CitationExtractor()
        text = "First point [Chunk: doc1#c0] second point [Chunk: doc2#c3]."
        ids = extractor.extract(text)
        assert "doc1#c0" in ids
        assert "doc2#c3" in ids

    def test_no_citations_returns_empty(self):
        extractor = CitationExtractor()
        assert extractor.extract("No citations here.") == []

    def test_whitespace_in_chunk_id(self):
        extractor = CitationExtractor()
        text = "[Chunk:  doc1#c0 ]"
        ids = extractor.extract(text)
        # The regex is non-greedy — whitespace is included in the ID as-is
        assert len(ids) == 1


# ---------------------------------------------------------------------------
# CitationVerifier unit tests
# ---------------------------------------------------------------------------


class TestCitationVerifier:
    """Unit tests for citation verification."""

    def test_all_valid_citations(self):
        verifier = CitationVerifier()
        result = verifier.verify(["c0", "c1"], {"c0": "text a", "c1": "text b"})
        assert result.is_verified is True
        assert result.hallucinated_citations == []
        assert result.verification_score == pytest.approx(1.0)

    def test_hallucinated_citation_detected(self):
        verifier = CitationVerifier()
        result = verifier.verify(["c0", "ghost"], {"c0": "text a"})
        assert result.is_verified is False
        assert "ghost" in result.hallucinated_citations
        assert result.verification_score == pytest.approx(0.5)

    def test_empty_citations(self):
        verifier = CitationVerifier()
        result = verifier.verify([], {"c0": "text"})
        assert result.is_verified is True
        assert result.verification_score == pytest.approx(1.0)

    def test_all_hallucinated(self):
        verifier = CitationVerifier()
        result = verifier.verify(["ghost1", "ghost2"], {})
        assert result.is_verified is False
        assert result.verification_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ConfidenceScorer unit tests
# ---------------------------------------------------------------------------


class TestConfidenceScorer:
    """Unit tests for retrieval-based confidence scoring."""

    def test_high_scores_no_abstain(self):
        scorer = ConfidenceScorer(threshold=0.7)
        result = scorer.evaluate("A normal answer.", [0.9, 0.85, 0.92])
        assert result.should_abstain is False
        assert result.confidence_score == pytest.approx(0.89, abs=0.01)

    def test_low_scores_abstain(self):
        scorer = ConfidenceScorer(threshold=0.7)
        result = scorer.evaluate("A normal answer.", [0.3, 0.4])
        assert result.should_abstain is True

    def test_i_dont_know_phrase_abstains(self):
        scorer = ConfidenceScorer(threshold=0.7)
        result = scorer.evaluate("I don't know based on the provided documents.", [0.9])
        assert result.should_abstain is True

    def test_empty_scores_abstains(self):
        scorer = ConfidenceScorer(threshold=0.7)
        result = scorer.evaluate("Some answer.", [])
        assert result.should_abstain is True
        assert result.confidence_score == pytest.approx(0.0)

    def test_custom_threshold(self):
        scorer = ConfidenceScorer(threshold=0.5)
        # Score of 0.6 is above 0.5 threshold
        result = scorer.evaluate("Answer.", [0.6])
        assert result.should_abstain is False
