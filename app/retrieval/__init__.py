"""Retrieval package supporting dense, sparse, hybrid RRF, and reranking."""

from app.retrieval.dense import DenseRetriever, RetrievalResult
from app.retrieval.sparse import BM25SparseStore, BM25IndexError, tokenize
from app.retrieval.fusion import ReciprocalRankFusion
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import VectorStore, VectorStoreError
from app.retrieval.indexer import ChunkIndexer

# Backward-compat alias — the old stub was called SparseRetriever
SparseRetriever = BM25SparseStore

__all__ = [
    "DenseRetriever",
    "RetrievalResult",
    "BM25SparseStore",
    "BM25IndexError",
    "tokenize",
    "SparseRetriever",
    "ReciprocalRankFusion",
    "CrossEncoderReranker",
    "VectorStore",
    "VectorStoreError",
    "ChunkIndexer",
]
