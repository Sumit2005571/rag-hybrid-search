"""FastAPI route handlers for query and ingestion endpoints."""

from fastapi import APIRouter, HTTPException, status
from app import __version__
from app.api.schemas import (
    DocumentIngestRequest,
    DocumentIngestResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from app.config import get_settings
from app.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Health check endpoint returning service status."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=__version__,
        environment=settings.app_env,
    )


@router.post("/query", response_model=QueryResponse, tags=["Search & Generation"])
def query_rag(request: QueryRequest) -> QueryResponse:
    """Execute hybrid search and grounded RAG answer generation.

    Note: Pipeline orchestration will be wired in downstream milestones.
    """
    logger.info("Received query request: query=%s", request.query)
    return QueryResponse(
        query=request.query,
        answer="Pipeline initialization in progress. Ready for milestone implementation.",
        citations=[],
        confidence_score=1.0,
        abstained=False,
        retrieved_chunks=[],
    )


@router.post("/ingest", response_model=DocumentIngestResponse, status_code=status.HTTP_201_CREATED, tags=["Ingestion"])
def ingest_document(request: DocumentIngestRequest) -> DocumentIngestResponse:
    """Ingest, chunk, and index a document into dense and sparse stores.

    Note: Implementation will be hooked up in the ingestion milestone.
    """
    logger.info("Received document ingest request: %s", request.source_path)
    return DocumentIngestResponse(
        document_id="doc_placeholder",
        total_chunks=0,
        status="queued",
    )
