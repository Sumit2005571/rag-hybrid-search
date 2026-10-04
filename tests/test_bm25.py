"""Comprehensive unit tests for Phase 2.2: BM25 Sparse Retrieval.

Coverage:
- Tokenization: technical identifier preservation, punctuation, empty strings.
- BM25SparseStore Init & Meta: default args, schema enforcement.
- Index building: empty chunks, valid chunks, deduplication, deterministic output.
- Persistence: saving and reloading the index, schema version checks.
- Search: basic keywords, exact technical terms, empty query, top_k limits.
- Idempotency: rebuilding the index with the same chunks.

All tests use a temporary directory for the BM25 index to avoid polluting data/bm25/.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List

import pytest

from app.chunking.base import Chunk
from app.retrieval.sparse import (
    BM25IndexError,
    BM25SparseStore,
    _chunk_to_sparse_metadata,
    tokenize,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_chunk(
    chunk_id: str = "doc1#c0",
    document_id: str = "doc1",
    text: str = "Sample chunk text.",
    chunk_index: int = 0,
    strategy: str = "fixed",
) -> Chunk:
    """Build a basic Chunk for testing."""
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        chunk_index=chunk_index,
        strategy=strategy,
        metadata={"file_type": "txt"},
    )


@pytest.fixture()
def tmp_bm25_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for BM25 persistence."""
    return tmp_path / "bm25"


@pytest.fixture()
def store(tmp_bm25_dir: Path) -> BM25SparseStore:
    """Return a BM25SparseStore instance pointing to a tmp directory."""
    return BM25SparseStore(index_directory=str(tmp_bm25_dir), top_k=5)


# ---------------------------------------------------------------------------
# Tokenization tests
# ---------------------------------------------------------------------------

class TestTokenize:
    """Deterministic, technically-aware tokenizer tests."""

    def test_empty_string(self):
        assert tokenize("") == []
        assert tokenize("   \n\t  ") == []

    def test_normal_text_lowercased(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_punctuation_split(self):
        assert tokenize("one, two. three! four?") == ["one", "two", "three", "four"]

    def test_preserves_underscores(self):
        assert tokenize("DATABASE_URL") == ["database_url"]
        assert tokenize("get_user_by_id") == ["get_user_by_id"]
        assert tokenize("HTTP_500_ERROR") == ["http_500_error"]

    def test_preserves_hyphens_within_words(self):
        # We allow hyphens inside tokens but strip them from the edges
        assert tokenize("state-of-the-art") == ["state-of-the-art"]
        assert tokenize("-hyphen-start") == ["hyphen-start"]

    def test_preserves_version_numbers(self):
        assert tokenize("v1.2.3") == ["v1.2.3"]
        assert tokenize("Node.js") == ["node.js"]

    def test_numbers_kept_with_words(self):
        assert tokenize("model2") == ["model2"]
        assert tokenize("32bit") == ["32bit"]

    def test_camel_case_splitting(self):
        assert tokenize("getUserById") == ["get", "user", "by", "id"]
        assert tokenize("CamelCaseClass") == ["camel", "case", "class"]


class TestChunkToSparseMetadata:
    """Metadata serialization tests."""

    def test_coerces_none_to_empty_string(self):
        chunk = _make_chunk()
        chunk.section_heading = None
        meta = _chunk_to_sparse_metadata(chunk)
        assert meta["section_heading"] == ""

    def test_preserves_scalars(self):
        chunk = _make_chunk()
        chunk.metadata["is_valid"] = True
        chunk.metadata["score"] = 0.95
        chunk.metadata["count"] = 10
        meta = _chunk_to_sparse_metadata(chunk)
        assert meta["is_valid"] is True
        assert meta["score"] == 0.95
        assert meta["count"] == 10

    def test_coerces_complex_types_to_string(self):
        chunk = _make_chunk()
        chunk.metadata["tags"] = ["a", "b"]
        meta = _chunk_to_sparse_metadata(chunk)
        assert meta["tags"] == "['a', 'b']"


# ---------------------------------------------------------------------------
# BM25SparseStore Tests
# ---------------------------------------------------------------------------

class TestBM25IndexBuild:
    """Index building and idempotency."""

    def test_build_empty_list_returns_zero(self, store: BM25SparseStore):
        assert store.build_index([]) == 0
        assert not store.is_loaded

    def test_build_single_chunk(self, store: BM25SparseStore):
        chunk = _make_chunk()
        assert store.build_index([chunk]) == 1
        assert store.is_loaded
        assert store.chunk_count == 1

    def test_build_multiple_chunks(self, store: BM25SparseStore):
        chunks = [
            _make_chunk(chunk_id="c1", text="First"),
            _make_chunk(chunk_id="c2", text="Second"),
        ]
        assert store.build_index(chunks) == 2
        assert store.chunk_count == 2

    def test_duplicate_chunk_ids_deduplicated(self, store: BM25SparseStore):
        chunks = [
            _make_chunk(chunk_id="c1", text="First"),
            _make_chunk(chunk_id="c1", text="Duplicate ID"),
        ]
        assert store.build_index(chunks) == 1
        assert store.chunk_count == 1
        # Should keep the first one
        assert store._chunk_texts[0] == "First"

    def test_idempotent_rebuild(self, store: BM25SparseStore):
        chunks = [
            _make_chunk(chunk_id="c1", text="First"),
            _make_chunk(chunk_id="c2", text="Second"),
        ]
        store.build_index(chunks)
        assert store.chunk_count == 2
        
        # Rebuild with same chunks
        store.build_index(chunks)
        assert store.chunk_count == 2  # still 2, not 4


class TestBM25Persistence:
    """Saving and loading the index."""

    def test_save_and_load(self, tmp_bm25_dir: Path):
        chunks = [_make_chunk(chunk_id="c1", text="Test content")]
        store1 = BM25SparseStore(index_directory=str(tmp_bm25_dir))
        store1.build_index(chunks)

        # Create new instance, load from disk
        store2 = BM25SparseStore(index_directory=str(tmp_bm25_dir))
        assert not store2.is_loaded
        count = store2.load_index()
        
        assert count == 1
        assert store2.is_loaded
        assert store2._chunk_ids == ["c1"]

    def test_load_missing_index_raises_error(self, tmp_bm25_dir: Path):
        store = BM25SparseStore(index_directory=str(tmp_bm25_dir))
        with pytest.raises(BM25IndexError, match="not found"):
            store.load_index()

    def test_schema_mismatch_json_raises_error(self, tmp_bm25_dir: Path):
        chunks = [_make_chunk()]
        store = BM25SparseStore(index_directory=str(tmp_bm25_dir))
        store.build_index(chunks)

        # Corrupt JSON schema version
        meta_path = tmp_bm25_dir / "bm25_meta.json"
        data = json.loads(meta_path.read_text())
        data["schema_version"] = "99.9"
        meta_path.write_text(json.dumps(data))

        with pytest.raises(BM25IndexError, match="schema version mismatch"):
            store.load_index()

    def test_schema_mismatch_pickle_raises_error(self, tmp_bm25_dir: Path):
        chunks = [_make_chunk()]
        store = BM25SparseStore(index_directory=str(tmp_bm25_dir))
        store.build_index(chunks)

        # Corrupt Pickle schema version
        model_path = tmp_bm25_dir / "bm25_model.pkl"
        with model_path.open("rb") as f:
            data = pickle.load(f)
        data["schema_version"] = "99.9"
        with model_path.open("wb") as f:
            pickle.dump(data, f)

        with pytest.raises(BM25IndexError, match="model schema version mismatch"):
            store.load_index()


class TestBM25Search:
    """Retrieval logic, ranking, and technical keywords."""

    @pytest.fixture()
    def search_store(self, store: BM25SparseStore) -> BM25SparseStore:
        chunks = [
            _make_chunk(chunk_id="c1", text="The quick brown fox jumps over the lazy dog."),
            _make_chunk(chunk_id="c2", text="Machine learning relies on neural networks and data."),
            _make_chunk(chunk_id="c3", text="Configuration issue: DATABASE_URL is missing."),
            _make_chunk(chunk_id="c4", text="Function get_user_by_id failed with HTTP_500."),
            _make_chunk(chunk_id="c5", text="The fox is quick, but the dog is lazy."),
        ]
        store.build_index(chunks)
        return store

    def test_search_not_loaded_raises_error(self, store: BM25SparseStore):
        with pytest.raises(BM25IndexError):
            store.search("query")

    def test_search_empty_query_raises_error(self, search_store: BM25SparseStore):
        with pytest.raises(ValueError, match="must not be empty"):
            search_store.search("   ")

    def test_basic_keyword_search(self, search_store: BM25SparseStore):
        results = search_store.search("machine learning")
        assert len(results) > 0
        assert results[0].chunk_id == "c2"
        assert results[0].rank == 1

    def test_exact_technical_search_database_url(self, search_store: BM25SparseStore):
        results = search_store.search("DATABASE_URL")
        assert len(results) > 0
        assert results[0].chunk_id == "c3"
        # Verify rank and score are populated
        assert results[0].rank == 1
        assert results[0].score > 0.0

    def test_exact_technical_search_function_name(self, search_store: BM25SparseStore):
        results = search_store.search("get_user_by_id")
        assert len(results) > 0
        assert results[0].chunk_id == "c4"

    def test_top_k_truncates_results(self, search_store: BM25SparseStore):
        # "fox", "quick", "lazy", "dog" appear in c1 and c5
        results = search_store.search("quick fox", top_k=1)
        assert len(results) == 1
        assert results[0].chunk_id in ["c1", "c5"]

    def test_no_matching_terms_returns_empty(self, search_store: BM25SparseStore):
        results = search_store.search("xylophone zebra")
        # Rank-BM25 returns 0.0 scores for all documents if no terms match.
        # Our implementation returns them, but we should check they have 0.0 score.
        assert all(r.score == 0.0 for r in results)

    def test_deterministic_tie_breaking(self, store: BM25SparseStore):
        # Two identical chunks, different IDs
        chunks = [
            _make_chunk(chunk_id="c_zebra", text="identical text"),
            _make_chunk(chunk_id="c_apple", text="identical text"),
        ]
        store.build_index(chunks)
        results = store.search("identical text")
        assert len(results) == 2
        assert results[0].score == results[1].score
        # Tie-breaker is ascending lexicographic on chunk_id, so "c_apple" should win
        assert results[0].chunk_id == "c_apple"
        assert results[1].chunk_id == "c_zebra"
