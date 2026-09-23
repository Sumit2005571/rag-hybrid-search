"""Application configuration module using environment variables."""

import os
from functools import lru_cache
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file if present
load_dotenv()


class Settings(BaseModel):
    """Application settings schema and defaults."""

    # Project metadata
    app_name: str = Field(default="RAG Hybrid Search", description="Name of the application")
    app_env: str = Field(default="development", description="Application environment: development, staging, production")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Global logging level")

    # API configuration
    api_host: str = Field(default="0.0.0.0", description="API host to bind to")
    api_port: int = Field(default=8000, description="API port to bind to")

    # OpenAI configuration
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name"
    )
    openai_chat_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI chat completion model name"
    )

    # Storage paths
    data_dir: str = Field(default="./data", description="Base data directory")
    chroma_persist_directory: str = Field(
        default="./data/chroma",
        description="Local directory for ChromaDB persistence"
    )
    chroma_collection_name: str = Field(
        default="rag_documents",
        description="ChromaDB collection name"
    )

    # Chunking defaults
    default_chunk_size: int = Field(default=500, description="Default chunk size in characters/tokens")
    default_chunk_overlap: int = Field(default=50, description="Default chunk overlap in characters/tokens")

    # Retrieval defaults
    top_k_dense: int = Field(default=10, description="Number of results from dense retrieval")
    top_k_sparse: int = Field(default=10, description="Number of results from sparse BM25 retrieval")
    top_k_final: int = Field(default=5, description="Final number of reranked results")
    rrf_k: int = Field(default=60, description="Reciprocal Rank Fusion k constant")
    cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking"
    )

    # Generation & Evaluation defaults
    confidence_threshold: float = Field(
        default=0.7,
        description="Threshold below which generation should abstain"
    )

    @classmethod
    def from_env(cls) -> "Settings":
        """Instantiate settings populated from environment variables."""
        return cls(
            app_name=os.getenv("APP_NAME", "RAG Hybrid Search"),
            app_env=os.getenv("APP_ENV", "development"),
            debug=os.getenv("DEBUG", "false").lower() in ("true", "1", "yes"),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            api_host=os.getenv("API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("API_PORT", "8000")),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            data_dir=os.getenv("DATA_DIR", "./data"),
            chroma_persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma"),
            chroma_collection_name=os.getenv("CHROMA_COLLECTION_NAME", "rag_documents"),
            default_chunk_size=int(os.getenv("DEFAULT_CHUNK_SIZE", "500")),
            default_chunk_overlap=int(os.getenv("DEFAULT_CHUNK_OVERLAP", "50")),
            top_k_dense=int(os.getenv("TOP_K_DENSE", "10")),
            top_k_sparse=int(os.getenv("TOP_K_SPARSE", "10")),
            top_k_final=int(os.getenv("TOP_K_FINAL", "5")),
            rrf_k=int(os.getenv("RRF_K", "60")),
            cross_encoder_model=os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.7")),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings.from_env()
