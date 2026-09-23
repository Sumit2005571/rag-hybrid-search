"""Unit tests for FastAPI endpoints."""

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


def test_query_endpoint(test_client: TestClient):
    """Verify query endpoint returns 200 with structured response."""
    response = test_client.post(
        "/api/v1/query",
        json={"query": "What is hybrid search?", "top_k": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "What is hybrid search?"
    assert "answer" in data
    assert "citations" in data


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
