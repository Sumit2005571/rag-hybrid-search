"""Sparse keyword retrieval using BM25."""

from typing import List
from app.retrieval.dense import RetrievalResult


class SparseRetriever:
    """Manages sparse BM25 lexical search."""

    def __init__(self):
        self.corpus: List[str] = []

    def fit(self, corpus: List[str]) -> None:
        """Tokenize corpus and initialize BM25 index."""
        self.corpus = corpus

    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Perform BM25 search for the query text.

        Note: BM25 index and tokenization will be implemented in sparse retrieval milestone.
        """
        return []
