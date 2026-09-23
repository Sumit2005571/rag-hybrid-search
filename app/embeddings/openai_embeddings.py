"""OpenAI embedding generator module."""

from typing import List, Optional
from app.config import get_settings


class OpenAIEmbeddingGenerator:
    """Generates dense vector embeddings using OpenAI text-embedding models."""

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.openai_embedding_model
        self.api_key = api_key or settings.openai_api_key

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of text strings.

        Note: Core API integration will be implemented in embeddings milestone.
        """
        if not texts:
            return []
        # Stub implementation for foundation milestone
        return [[0.0] * 1536 for _ in texts]

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding vector for a single query string."""
        return self.embed_texts([query])[0]
