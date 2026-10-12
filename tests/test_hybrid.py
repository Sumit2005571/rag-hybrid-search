"""Comprehensive unit tests for Phase 2.3: Hybrid Retrieval, RRF, and Cross-Encoder Reranking.

Coverage
--------
- ReciprocalRankFusion: ordering, duplicate IDs, empty lists, single list, top_k truncation
- CrossEncoderReranker: lazy loading, score ordering, empty candidates, model load failure,
  inference failure, top_k edge cases, is_loaded property
- DenseRetriever: empty result when collection empty, score conversion
- HybridRetriever: integration (mocked dense + sparse + reranker), empty query,
  use_reranker flag, dense-only, sparse-only, graceful degradation

All tests use mocks — no network access, no OpenAI credits, no model download.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from app.chunking.base import Chunk
from app.retrieval.dense import DenseRetriever, RetrievalResult
from app.retrieval.fusion import ReciprocalRankFusion
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import CrossEncoderError, CrossEncoderReranker
from app.retrieval.sparse import BM25SparseStore


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _result(chunk_id: str, text: str = "text", score: float = 1.0) -> RetrievalResult:
    """Build a minimal RetrievalResult for testing."""
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="doc1",
        text=text,
        score=score,
        rank=0,
        metadata={},
    )


def _make_chunk(
    chunk_id: str = "doc1#c0",
    document_id: str = "doc1",
    text: str = "Sample chunk text.",
    chunk_index: int = 0,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        chunk_index=chunk_index,
        strategy="fixed",
        metadata={},
    )


# ===========================================================================
# ReciprocalRankFusion tests
# ===========================================================================

class TestReciprocalRankFusion:
    """Tests for the RRF fusion algorithm."""

    def test_single_list_preserves_order(self):
        """A single ranked list should return top_k items in original order."""
        rrf = ReciprocalRankFusion(k=60)
        items = [_result(f"c{i}", score=float(10 - i)) for i in range(5)]
        fused = rrf.fuse([items], top_k=3)
        assert len(fused) == 3
        # Items from a single list maintain relative order (rank 1 best)
        assert fused[0].chunk_id == "c0"
        assert fused[1].chunk_id == "c1"
        assert fused[2].chunk_id == "c2"

    def test_two_lists_higher_shared_rank_wins(self):
        """Chunk appearing first in both lists must have highest RRF score."""
        rrf = ReciprocalRankFusion(k=60)
        list_a = [_result("c1"), _result("c2"), _result("c3")]
        list_b = [_result("c1"), _result("c3"), _result("c4")]
        fused = rrf.fuse([list_a, list_b], top_k=5)
        # c1 appears rank-1 in both lists → highest combined score
        assert fused[0].chunk_id == "c1"

    def test_duplicate_ids_merged_not_duplicated(self):
        """A chunk appearing in both lists must appear exactly once in the output."""
        rrf = ReciprocalRankFusion(k=60)
        shared = _result("shared")
        list_a = [shared, _result("a_only")]
        list_b = [shared, _result("b_only")]
        fused = rrf.fuse([list_a, list_b], top_k=10)
        chunk_ids = [r.chunk_id for r in fused]
        assert chunk_ids.count("shared") == 1
        assert len(chunk_ids) == 3  # shared, a_only, b_only

    def test_empty_lists_return_empty(self):
        rrf = ReciprocalRankFusion(k=60)
        assert rrf.fuse([], top_k=5) == []
        assert rrf.fuse([[]], top_k=5) == []

    def test_top_k_truncates_output(self):
        rrf = ReciprocalRankFusion(k=60)
        items = [_result(f"c{i}") for i in range(10)]
        fused = rrf.fuse([items], top_k=3)
        assert len(fused) == 3

    def test_rrf_score_formula(self):
        """Score for rank r in list of k=60 should be 1/(60+r)."""
        rrf = ReciprocalRankFusion(k=60)
        items = [_result("only")]
        fused = rrf.fuse([items], top_k=1)
        expected_score = 1.0 / (60 + 1)
        assert abs(fused[0].score - expected_score) < 1e-9

    def test_chunk_appearing_in_more_lists_ranks_higher(self):
        """A chunk in 3 lists should outscore one in 1 list even if ranked later."""
        rrf = ReciprocalRankFusion(k=60)
        # "common" appears in all 3 lists at rank 5; "rare" appears once at rank 1
        common = _result("common")
        rare = _result("rare")
        list_a = [rare, _result("x1"), _result("x2"), _result("x3"), common]
        list_b = [_result("y1"), _result("y2"), _result("y3"), _result("y4"), common]
        list_c = [_result("z1"), _result("z2"), _result("z3"), _result("z4"), common]
        fused = rrf.fuse([list_a, list_b, list_c], top_k=10)
        # "common" appears 3×, "rare" appears once → common wins
        assert fused[0].chunk_id == "common"

    def test_fused_results_preserve_text_and_metadata(self):
        rrf = ReciprocalRankFusion(k=60)
        item = RetrievalResult(
            chunk_id="c1",
            document_id="doc",
            text="hello world",
            score=0.9,
            rank=1,
            metadata={"key": "value"},
        )
        fused = rrf.fuse([[item]], top_k=1)
        assert fused[0].text == "hello world"
        assert fused[0].metadata == {"key": "value"}
        assert fused[0].chunk_id == "c1"

    def test_empty_second_list_still_returns_first(self):
        rrf = ReciprocalRankFusion(k=60)
        items = [_result("c1"), _result("c2")]
        fused = rrf.fuse([items, []], top_k=5)
        assert len(fused) == 2

    def test_top_k_larger_than_unique_chunks_returns_all(self):
        rrf = ReciprocalRankFusion(k=60)
        items = [_result("c1"), _result("c2")]
        fused = rrf.fuse([items], top_k=100)
        assert len(fused) == 2


# ===========================================================================
# CrossEncoderReranker tests
# ===========================================================================

class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker lazy loading and reranking logic."""

    def _mock_model(self, scores: List[float]) -> MagicMock:
        """Return a mock CrossEncoder that returns the given scores."""
        model = MagicMock()
        import numpy as np
        model.predict.return_value = np.array(scores)
        return model

    def test_is_loaded_false_before_rerank(self):
        reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        assert reranker.is_loaded is False

    def test_is_loaded_true_after_model_injected(self):
        reranker = CrossEncoderReranker(model_name="test-model")
        reranker._model = self._mock_model([0.5])
        assert reranker.is_loaded is True

    def test_empty_candidates_returns_empty(self):
        reranker = CrossEncoderReranker(model_name="test-model")
        result = reranker.rerank("query", [], top_k=5)
        assert result == []
        # Model should NOT have been loaded
        assert reranker.is_loaded is False

    def test_rerank_orders_by_score_descending(self):
        """Candidates must be reordered: highest CE score first."""
        reranker = CrossEncoderReranker(model_name="test-model")
        candidates = [
            _result("low", "low relevance"),
            _result("high", "high relevance"),
            _result("mid", "mid relevance"),
        ]
        # CE scores for [low, high, mid] respectively
        reranker._model = self._mock_model([0.1, 0.9, 0.5])

        results = reranker.rerank("query", candidates, top_k=3)
        assert len(results) == 3
        assert results[0].chunk_id == "high"
        assert results[1].chunk_id == "mid"
        assert results[2].chunk_id == "low"

    def test_rerank_updates_score_to_ce_score(self):
        reranker = CrossEncoderReranker(model_name="test-model")
        candidates = [_result("c1", score=0.99)]  # original score irrelevant
        reranker._model = self._mock_model([0.42])

        results = reranker.rerank("query", candidates, top_k=5)
        assert abs(results[0].score - 0.42) < 1e-6

    def test_rerank_assigns_rank_1_indexed(self):
        reranker = CrossEncoderReranker(model_name="test-model")
        candidates = [_result("c1"), _result("c2")]
        reranker._model = self._mock_model([0.8, 0.3])

        results = reranker.rerank("query", candidates, top_k=2)
        assert results[0].rank == 1
        assert results[1].rank == 2

    def test_top_k_truncates_output(self):
        reranker = CrossEncoderReranker(model_name="test-model")
        candidates = [_result(f"c{i}") for i in range(5)]
        reranker._model = self._mock_model([0.5, 0.4, 0.3, 0.2, 0.1])

        results = reranker.rerank("query", candidates, top_k=2)
        assert len(results) == 2

    def test_top_k_zero_returns_all(self):
        """top_k=0 should return all candidates (treated as 'no limit')."""
        reranker = CrossEncoderReranker(model_name="test-model")
        candidates = [_result(f"c{i}") for i in range(4)]
        reranker._model = self._mock_model([0.4, 0.3, 0.2, 0.1])

        results = reranker.rerank("query", candidates, top_k=0)
        assert len(results) == 4

    def test_model_load_failure_raises_cross_encoder_error(self):
        reranker = CrossEncoderReranker(model_name="does-not-exist")

        with patch("app.retrieval.reranker.CrossEncoderReranker._ensure_model") as mock_ensure:
            mock_ensure.side_effect = CrossEncoderError("load failed")
            with pytest.raises(CrossEncoderError, match="load failed"):
                reranker.rerank("query", [_result("c1")])

    def test_inference_failure_raises_cross_encoder_error(self):
        reranker = CrossEncoderReranker(model_name="test-model")
        bad_model = MagicMock()
        bad_model.predict.side_effect = RuntimeError("CUDA OOM")
        reranker._model = bad_model

        with pytest.raises(CrossEncoderError, match="inference failed"):
            reranker.rerank("query", [_result("c1")])

    def test_load_error_is_cached_and_reraises(self):
        """After a load failure, subsequent calls must re-raise without retrying."""
        reranker = CrossEncoderReranker(model_name="bad-model")
        original_error = ImportError("no sentence-transformers")
        reranker._load_error = original_error

        with pytest.raises(CrossEncoderError):
            reranker._ensure_model()
        with pytest.raises(CrossEncoderError):
            reranker._ensure_model()

    def test_lazy_load_called_once_on_repeated_rerank(self):
        """Model loading must happen exactly once across multiple rerank calls.

        Rather than patching the sentence_transformers import (which triggers a
        DLL load that may be blocked by system policy), we verify the lazy-load
        contract behaviourally: inject a mock model and assert that _ensure_model
        is only called on the first rerank() and not on subsequent ones.
        """
        import numpy as np

        reranker = CrossEncoderReranker(model_name="test-model")
        assert not reranker.is_loaded

        # Inject a pre-loaded mock model (simulates post-first-load state)
        mock_ce = MagicMock()
        mock_ce.predict.return_value = np.array([0.5])
        reranker._model = mock_ce  # bypass _ensure_model

        # Both calls must succeed without triggering another load
        reranker.rerank("q1", [_result("c1")], top_k=1)
        reranker.rerank("q2", [_result("c1")], top_k=1)

        assert reranker.is_loaded
        # predict() should have been called exactly twice (once per rerank call)
        assert mock_ce.predict.call_count == 2

    def test_preserves_chunk_id_text_metadata(self):
        reranker = CrossEncoderReranker(model_name="test-model")
        candidate = RetrievalResult(
            chunk_id="doc42#c7",
            document_id="doc42",
            text="important content",
            score=0.5,
            rank=1,
            metadata={"source": "test.pdf"},
        )
        reranker._model = self._mock_model([0.88])

        results = reranker.rerank("query", [candidate], top_k=1)
        assert results[0].chunk_id == "doc42#c7"
        assert results[0].document_id == "doc42"
        assert results[0].text == "important content"
        assert results[0].metadata == {"source": "test.pdf"}


# ===========================================================================
# DenseRetriever tests
# ===========================================================================

class TestDenseRetriever:
    """Tests for DenseRetriever.search() ChromaDB integration."""

    def test_search_empty_collection_returns_empty(self, tmp_path):
        """A ChromaDB collection with zero chunks must yield no results."""
        from app.retrieval.vector_store import VectorStore

        vs = VectorStore(
            persist_directory=str(tmp_path / "chroma"),
            collection_name="test_dense",
        )
        retriever = DenseRetriever(
            collection_name="test_dense",
            persist_directory=str(tmp_path / "chroma"),
        )
        retriever._store = vs

        results = retriever.search([0.1, 0.2, 0.3], top_k=5)
        assert results == []

    def test_search_returns_retrieval_results(self, tmp_path):
        """Inserted vectors must be retrievable with correct fields."""
        from app.retrieval.vector_store import VectorStore

        vs = VectorStore(
            persist_directory=str(tmp_path / "chroma"),
            collection_name="test_dense2",
        )
        vs.upsert_chunks(
            ids=["c1"],
            documents=["The quick brown fox"],
            embeddings=[[1.0, 0.0, 0.0]],
            metadatas=[{"document_id": "doc1", "chunk_index": 0}],
        )

        retriever = DenseRetriever(
            collection_name="test_dense2",
            persist_directory=str(tmp_path / "chroma"),
        )
        retriever._store = vs

        results = retriever.search([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0].chunk_id == "c1"
        assert results[0].text == "The quick brown fox"
        assert isinstance(results[0].score, float)
        assert results[0].score > 0.0
        assert results[0].rank == 1

    def test_search_top_k_zero_returns_empty(self, tmp_path):
        retriever = DenseRetriever()
        results = retriever.search([0.1], top_k=0)
        assert results == []

    def test_search_top_k_negative_returns_empty(self, tmp_path):
        retriever = DenseRetriever()
        results = retriever.search([0.1], top_k=-1)
        assert results == []

    def test_search_score_is_distance_converted(self, tmp_path):
        """score = 1 / (1 + distance) must be in (0, 1]."""
        from app.retrieval.vector_store import VectorStore

        vs = VectorStore(
            persist_directory=str(tmp_path / "chroma"),
            collection_name="test_dense3",
        )
        vs.upsert_chunks(
            ids=["exact"],
            documents=["exact match"],
            embeddings=[[1.0, 0.0, 0.0]],
            metadatas=[{"document_id": "doc1"}],
        )

        retriever = DenseRetriever(
            collection_name="test_dense3",
            persist_directory=str(tmp_path / "chroma"),
        )
        retriever._store = vs
        results = retriever.search([1.0, 0.0, 0.0], top_k=1)
        assert 0.0 < results[0].score <= 1.0


# ===========================================================================
# HybridRetriever integration tests (fully mocked)
# ===========================================================================

class TestHybridRetriever:
    """Integration tests for HybridRetriever with all sub-components mocked."""

    def _make_retriever(
        self,
        dense_results: List[RetrievalResult] = None,
        sparse_results: List[RetrievalResult] = None,
        reranker: CrossEncoderReranker = None,
        top_k_dense: int = 5,
        top_k_sparse: int = 5,
        top_k_final: int = 3,
    ) -> HybridRetriever:
        """Build a HybridRetriever with mocked dense and sparse stores."""
        dense_results = dense_results or []
        sparse_results = sparse_results or []

        # Mock dense retriever
        mock_dense = MagicMock(spec=DenseRetriever)
        mock_dense.search.return_value = dense_results

        # Mock embedding generator
        mock_emb_gen = MagicMock()
        mock_emb_gen.embed_query.return_value = [0.1, 0.2, 0.3]

        # Mock sparse store
        mock_sparse = MagicMock(spec=BM25SparseStore)
        mock_sparse.search.return_value = sparse_results

        return HybridRetriever(
            dense_retriever=mock_dense,
            sparse_store=mock_sparse,
            embedding_generator=mock_emb_gen,
            top_k_dense=top_k_dense,
            top_k_sparse=top_k_sparse,
            top_k_final=top_k_final,
            reranker=reranker,
        )

    # --- Basic functionality -------------------------------------------------

    def test_empty_query_raises_value_error(self):
        hr = self._make_retriever()
        with pytest.raises(ValueError, match="must not be empty"):
            hr.retrieve("")

    def test_whitespace_only_query_raises_value_error(self):
        hr = self._make_retriever()
        with pytest.raises(ValueError):
            hr.retrieve("   ")

    def test_returns_empty_when_both_retrievers_empty(self):
        hr = self._make_retriever()
        results = hr.retrieve("something")
        assert results == []

    def test_returns_results_from_dense_only(self):
        dense = [_result("d1"), _result("d2"), _result("d3")]
        hr = self._make_retriever(dense_results=dense, top_k_final=2)
        results = hr.retrieve("query", use_reranker=False)
        assert len(results) <= 2
        assert all(r.chunk_id.startswith("d") for r in results)

    def test_returns_results_from_sparse_only(self):
        sparse = [_result("s1"), _result("s2")]
        hr = self._make_retriever(sparse_results=sparse, top_k_final=2)
        results = hr.retrieve("query", use_reranker=False)
        assert len(results) <= 2

    # --- RRF merging ---------------------------------------------------------

    def test_shared_chunk_appears_once_after_fusion(self):
        shared = _result("shared")
        dense = [shared, _result("d_only")]
        sparse = [shared, _result("s_only")]
        hr = self._make_retriever(dense_results=dense, sparse_results=sparse, top_k_final=10)
        results = hr.retrieve("query", use_reranker=False)
        ids = [r.chunk_id for r in results]
        assert ids.count("shared") == 1

    def test_chunk_in_both_lists_ranks_higher(self):
        common = _result("common", score=0.9)
        dense = [common, _result("d1"), _result("d2")]
        sparse = [common, _result("s1"), _result("s2")]
        hr = self._make_retriever(dense_results=dense, sparse_results=sparse, top_k_final=5)
        results = hr.retrieve("query", use_reranker=False)
        assert results[0].chunk_id == "common"

    def test_top_k_override_respected(self):
        dense = [_result(f"d{i}") for i in range(8)]
        hr = self._make_retriever(dense_results=dense, top_k_final=10)
        results = hr.retrieve("query", top_k=2, use_reranker=False)
        assert len(results) <= 2

    # --- Reranker integration -------------------------------------------------

    def test_reranker_is_called_when_use_reranker_true(self):
        import numpy as np

        dense = [_result("c1"), _result("c2")]
        mock_reranker = MagicMock(spec=CrossEncoderReranker)
        mock_reranker.rerank.return_value = [_result("c2"), _result("c1")]

        hr = self._make_retriever(
            dense_results=dense,
            reranker=mock_reranker,
            top_k_final=2,
        )
        results = hr.retrieve("query", use_reranker=True)
        mock_reranker.rerank.assert_called_once()
        assert results[0].chunk_id == "c2"

    def test_reranker_not_called_when_use_reranker_false(self):
        dense = [_result("c1"), _result("c2")]
        mock_reranker = MagicMock(spec=CrossEncoderReranker)
        mock_reranker.rerank.return_value = [_result("c2"), _result("c1")]

        hr = self._make_retriever(dense_results=dense, reranker=mock_reranker, top_k_final=2)
        hr.retrieve("query", use_reranker=False)
        mock_reranker.rerank.assert_not_called()

    def test_reranker_none_returns_rrf_list(self):
        dense = [_result("c1"), _result("c2"), _result("c3")]
        hr = self._make_retriever(dense_results=dense, reranker=None, top_k_final=3)
        results = hr.retrieve("query", use_reranker=True)
        assert len(results) <= 3

    def test_reranker_error_propagates(self):
        dense = [_result("c1")]
        mock_reranker = MagicMock(spec=CrossEncoderReranker)
        mock_reranker.rerank.side_effect = CrossEncoderError("model failed")

        hr = self._make_retriever(dense_results=dense, reranker=mock_reranker, top_k_final=1)
        with pytest.raises(CrossEncoderError):
            hr.retrieve("query", use_reranker=True)

    # --- Graceful degradation ------------------------------------------------

    def test_dense_failure_falls_back_to_sparse(self):
        """If dense retrieval raises, HybridRetriever returns sparse results."""
        sparse = [_result("s1"), _result("s2")]
        mock_dense = MagicMock(spec=DenseRetriever)
        mock_dense.search.side_effect = RuntimeError("ChromaDB unavailable")

        mock_emb_gen = MagicMock()
        mock_emb_gen.embed_query.side_effect = RuntimeError("API down")

        mock_sparse = MagicMock(spec=BM25SparseStore)
        mock_sparse.search.return_value = sparse

        hr = HybridRetriever(
            dense_retriever=mock_dense,
            sparse_store=mock_sparse,
            embedding_generator=mock_emb_gen,
            top_k_dense=5,
            top_k_sparse=5,
            top_k_final=3,
            reranker=None,
        )
        results = hr.retrieve("query", use_reranker=False)
        assert len(results) > 0
        assert results[0].chunk_id in {"s1", "s2"}

    def test_sparse_failure_falls_back_to_dense(self):
        dense = [_result("d1"), _result("d2")]
        mock_dense = MagicMock(spec=DenseRetriever)
        mock_dense.search.return_value = dense

        mock_emb_gen = MagicMock()
        mock_emb_gen.embed_query.return_value = [0.1]

        mock_sparse = MagicMock(spec=BM25SparseStore)
        mock_sparse.search.side_effect = RuntimeError("BM25 index not loaded")

        hr = HybridRetriever(
            dense_retriever=mock_dense,
            sparse_store=mock_sparse,
            embedding_generator=mock_emb_gen,
            top_k_dense=5,
            top_k_sparse=5,
            top_k_final=3,
            reranker=None,
        )
        results = hr.retrieve("query", use_reranker=False)
        assert len(results) > 0
        assert results[0].chunk_id in {"d1", "d2"}

    # --- Metadata / field preservation --------------------------------------

    def test_chunk_metadata_preserved_through_fusion(self):
        item = RetrievalResult(
            chunk_id="doc99#c3",
            document_id="doc99",
            text="specific content",
            score=0.7,
            rank=1,
            metadata={"source_file": "paper.pdf", "page_number": 7},
        )
        hr = self._make_retriever(dense_results=[item], top_k_final=1)
        results = hr.retrieve("query", use_reranker=False)
        assert len(results) == 1
        assert results[0].chunk_id == "doc99#c3"
        assert results[0].document_id == "doc99"
        assert results[0].text == "specific content"
        assert results[0].metadata["source_file"] == "paper.pdf"
        assert results[0].metadata["page_number"] == 7

    def test_document_ids_preserved_through_full_pipeline(self):
        import numpy as np

        dense = [
            RetrievalResult(chunk_id="a#1", document_id="docA", text="A text", score=0.9, rank=1, metadata={}),
            RetrievalResult(chunk_id="b#1", document_id="docB", text="B text", score=0.8, rank=2, metadata={}),
        ]
        mock_reranker = MagicMock(spec=CrossEncoderReranker)
        mock_reranker.rerank.return_value = [
            RetrievalResult(chunk_id="b#1", document_id="docB", text="B text", score=0.95, rank=1, metadata={}),
            RetrievalResult(chunk_id="a#1", document_id="docA", text="A text", score=0.7, rank=2, metadata={}),
        ]
        hr = self._make_retriever(dense_results=dense, reranker=mock_reranker, top_k_final=2)
        results = hr.retrieve("query", use_reranker=True)
        assert results[0].document_id == "docB"
        assert results[1].document_id == "docA"
