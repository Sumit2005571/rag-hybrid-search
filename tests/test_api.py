"""Unit tests for FastAPI endpoints.

These tests mock all external dependencies (OpenAI, ChromaDB, BM25) so that
the suite runs without network access, API keys, or pre-built indexes.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


def test_health_check(test_client: TestClient):
    """Verify health check endpoint returns 200 and expected schema."""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "environment" in data


def test_api_v1_health(test_client: TestClient):
    """Verify prefixed health endpoint."""
    response = test_client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_endpoint_basic(test_client: TestClient):
    """Verify /query returns 200 with structured response when retriever is mocked."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    mock_gen_result = MagicMock()
    mock_gen_result.answer = "I don't know based on the provided documents."
    mock_gen_result.citations = []
    mock_gen_result.confidence.confidence_score = 0.0
    mock_gen_result.confidence.should_abstain = True
    mock_gen_result.verification = None

    with patch("app.api.routes._build_hybrid_retriever", return_value=mock_retriever), \
         patch("app.api.routes.RAGGenerator") as MockGen:
        MockGen.return_value.generate.return_value = mock_gen_result
        response = test_client.post(
            "/api/v1/query",
            json={"query": "What is hybrid search?", "top_k": 3},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "What is hybrid search?"
    assert "answer" in data
    assert isinstance(data["citations"], list)
    assert "abstained" in data
    assert "confidence_score" in data


def test_ingest_endpoint(test_client: TestClient):
    """Verify document ingest endpoint returns 201."""
    response = test_client.post(
        "/api/v1/ingest",
        json={"source_path": "data/raw/sample.txt", "chunking_strategy": "fixed"},
    )
    assert response.status_code == 201
    data = response.json()
    assert "document_id" in data
    assert data["status"] == "queued"
