"""Unit tests for configuration management."""

from app.config import Settings, get_settings


def test_default_settings():
    """Verify default configuration attributes."""
    settings = Settings.from_env()
    assert settings.app_name is not None
    assert settings.default_chunk_size > 0
    assert settings.default_chunk_overlap >= 0
    assert settings.top_k_dense > 0
    assert settings.top_k_sparse > 0
    assert settings.top_k_final > 0
    assert 0.0 <= settings.confidence_threshold <= 1.0


def test_settings_from_env(mock_env):
    """Verify settings correctly reflect environment variables."""
    settings = Settings.from_env()
    assert settings.app_name == "RAG Test Suite"
    assert settings.app_env == "test"
    assert settings.log_level == "DEBUG"
    assert settings.openai_api_key == "test-key-mock"
    assert settings.default_chunk_size == 200
    assert settings.default_chunk_overlap == 20


def test_get_settings_singleton():
    """Verify get_settings returns a cached instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
