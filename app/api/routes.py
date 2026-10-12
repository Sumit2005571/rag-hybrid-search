"""FastAPI route handlers for query and ingestion endpoints."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, status

from app import __version__
from app.api.schemas import (
    CitationItem,
    DocumentIngestRequest,
    DocumentIngestResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from app.config import get_settings
from app.generation.generator import GenerationError, RAGGenerator
from app.logging_config import get_logger
from app.retrieval.dense import DenseRetriever, RetrievalResult

router = APIRouter()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Query (hybrid retrieval + grounded generation)
# ---------------------------------------------------------------------------


def _build_hybrid_retriever():
    """Construct a HybridRetriever from application settings.

    Raises:
        RuntimeError: If required dependencies (dense store, BM25 index, or
            embeddings) cannot be initialised.
    """
    # Deferred imports to keep startup cost low and allow tests to patch.
    from app.embeddings.embeddings import EmbeddingGenerator
    from app.retrieval.hybrid import HybridRetriever
    from app.retrieval.sparse import BM25SparseStore

    settings = get_settings()

    # The EmbeddingGenerator validates the API key at init time.
    embedding_gen = EmbeddingGenerator(validate_on_init=True)
    dense_ret = DenseRetriever()
    sparse_store = BM25SparseStore(index_directory=settings.bm25_index_directory)

    # Attempt to load the persisted BM25 index; log a warning if not found.
    try:
        sparse_store.load_index()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "BM25 index could not be loaded (sparse retrieval will return no results): %s", exc
        )

    return HybridRetriever(
        dense_retriever=dense_ret,
        sparse_store=sparse_store,
        embedding_generator=embedding_gen,
    )


@router.post("/query", response_model=QueryResponse, tags=["Search & Generation"])
def query_rag(request: QueryRequest) -> QueryResponse:
    """Execute hybrid search and grounded RAG answer generation.

    Pipeline
    --------
    1. Hybrid retrieval: dense (ChromaDB) + sparse (BM25) → RRF fusion → optional reranking.
    2. Grounded generation: OpenAI chat completion with retrieved contexts.
    3. Citation extraction from the generated answer and verification against retrieved chunks.
    4. Confidence assessment and abstention when retrieval quality is below threshold.

    Errors
    ------
    - 400: Empty query string.
    - 503: Required services unavailable (missing API key, uninitialised indexes).
    - 500: Unexpected internal error.
    """
    logger.info("Received query request: query=%r, top_k=%d", request.query, request.top_k or 5)

    # ---- Build retriever (may raise if API key missing) ---------------
    try:
        retriever = _build_hybrid_retriever()
    except Exception as exc:
        logger.error("Could not initialise HybridRetriever: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Retrieval service is unavailable. Ensure OPENAI_API_KEY is set and "
                "that the ChromaDB index has been populated via the ingestion script."
            ),
        ) from exc

    # ---- Hybrid retrieval ---------------------------------------------
    try:
        results: List[RetrievalResult] = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            use_reranker=request.use_reranker if request.use_reranker is not None else True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Hybrid retrieval failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval failed. Check that the vector and BM25 indexes are populated.",
        ) from exc

    logger.info("Retrieved %d chunks for query %r", len(results), request.query)

    # ---- Grounded generation ------------------------------------------
    generator = RAGGenerator()
    generation = generator.generate(question=request.query, retrieved_contexts=results)

    # ---- Build citation items for the response -----------------------
    # Build a lookup from chunk_id → RetrievalResult for metadata access.
    chunk_lookup: Dict[str, RetrievalResult] = {r.chunk_id: r for r in results}
    citation_items: List[CitationItem] = [
        CitationItem(
            chunk_id=cid,
            source=chunk_lookup[cid].metadata.get("source_file") or chunk_lookup[cid].document_id or None,
            text_snippet=chunk_lookup[cid].text[:200] if cid in chunk_lookup else None,
        )
        for cid in generation.citations
        if cid in chunk_lookup  # already filtered by CitationVerifier, but guard anyway
    ]

    # ---- Serialise retrieved chunks for inspection --------------------
    retrieved_chunks_payload: List[Dict[str, Any]] = [
        {
            "chunk_id": r.chunk_id,
            "document_id": r.document_id,
            "text": r.text,
            "score": r.score,
            "rank": r.rank,
            "metadata": r.metadata,
        }
        for r in results
    ]

    return QueryResponse(
        query=request.query,
        answer=generation.answer,
        citations=citation_items,
        confidence_score=generation.confidence.confidence_score,
        abstained=generation.confidence.should_abstain,
        retrieved_chunks=retrieved_chunks_payload,
        verification_score=(
            generation.verification.verification_score if generation.verification is not None else None
        ),
    )


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


@router.post(
    "/ingest",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Ingestion"],
)
def ingest_document(request: DocumentIngestRequest) -> DocumentIngestResponse:
    """Ingest, chunk, and index a document into dense and sparse stores.

    Note: Full implementation is wired in the ingestion milestone.
    """
    logger.info("Received document ingest request: %s", request.source_path)
    return DocumentIngestResponse(
        document_id="doc_placeholder",
        total_chunks=0,
        status="queued",
    )
