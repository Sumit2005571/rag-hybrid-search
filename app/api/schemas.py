"""FastAPI request and response schemas."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check status response."""

    status: str = Field(default="ok", description="Service health status")
    app_name: str = Field(..., description="Application name")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Running environment")


class QueryRequest(BaseModel):
    """User query request schema."""

    query: str = Field(..., min_length=1, description="User question or search query")
    top_k: Optional[int] = Field(default=5, ge=1, le=50, description="Number of results to return")
    use_reranker: Optional[bool] = Field(default=True, description="Enable cross-encoder reranking")


class CitationItem(BaseModel):
    """Citation details returned in query responses."""

    chunk_id: str
    source: Optional[str] = None
    text_snippet: Optional[str] = None


class QueryResponse(BaseModel):
    """Complete RAG answer response."""

    query: str
    answer: str
    citations: List[str]
    confidence_score: float
    abstained: bool
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)


class DocumentIngestRequest(BaseModel):
    """Document ingest payload schema."""

    source_path: str = Field(..., description="Path to document file to ingest")
    chunking_strategy: Optional[str] = Field(default="fixed", description="Strategy: fixed, recursive, semantic")


class DocumentIngestResponse(BaseModel):
    """Response returned upon document ingestion."""

    document_id: str
    total_chunks: int
    status: str
