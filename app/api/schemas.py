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

    chunk_id: str = Field(..., description="ID of the cited chunk")
    source: Optional[str] = Field(default=None, description="Source file or document")
    text_snippet: Optional[str] = Field(default=None, description="Excerpt from the cited chunk")


class QueryResponse(BaseModel):
    """Complete RAG answer response."""

    query: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Generated answer text")
    citations: List[CitationItem] = Field(
        default_factory=list,
        description="Citations for claims supported by retrieved chunks",
    )
    confidence_score: float = Field(
        ...,
        description=(
            "Average retrieval relevance score used for abstention decisions. "
            "This reflects retrieval quality, not a calibrated probability of correctness."
        ),
    )
    abstained: bool = Field(
        ...,
        description="True when the system chose not to answer due to low retrieval confidence",
    )
    retrieved_chunks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Retrieved context chunks (chunk_id, text, score, metadata)",
    )
    verification_score: Optional[float] = Field(
        default=None,
        description="Fraction of cited chunk IDs that were verified as present in the retrieved context",
    )


class DocumentIngestRequest(BaseModel):
    """Document ingest payload schema."""

    source_path: str = Field(..., description="Path to document file to ingest")
    chunking_strategy: Optional[str] = Field(default="fixed", description="Strategy: fixed, recursive, semantic")


class DocumentIngestResponse(BaseModel):
    """Response returned upon document ingestion."""

    document_id: str
    total_chunks: int
    status: str
