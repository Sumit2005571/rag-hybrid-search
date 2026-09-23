"""Shared pytest fixtures."""

import os
import pytest
from fastapi.testclient import TestClient
from app.config import Settings
from app.main import app


@pytest.fixture
def mock_env(monkeypatch):
    """Set standard testing environment variables."""
    monkeypatch.setenv("APP_NAME", "RAG Test Suite")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-mock")
    monkeypatch.setenv("DEFAULT_CHUNK_SIZE", "200")
    monkeypatch.setenv("DEFAULT_CHUNK_OVERLAP", "20")


@pytest.fixture
def test_settings(mock_env) -> Settings:
    """Return an isolated Settings instance for tests."""
    return Settings.from_env()


@pytest.fixture
def test_client() -> TestClient:
    """FastAPI TestClient fixture."""
    return TestClient(app)
