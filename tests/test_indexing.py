"""Comprehensive unit tests for Phase 2.1: Embeddings + ChromaDB indexing.

Coverage:
- EmbeddingGenerator: config loading, missing API key, batching, empty input
- VectorStore: init, collection creation/reopening, upsert, idempotency, metadata
- ChunkIndexer: indexing chunks, deterministic IDs, re-indexing, invalid data
- CLI helpers: load_chunks_from_directory

All external OpenAI API calls are mocked — no live API key required.
ChromaDB uses a temporary directory to avoid polluting data/chroma/.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from app.chunking.base import Chunk
from app.embeddings.embeddings import (
    EmbeddingAPIError,
    EmbeddingConfigError,
    EmbeddingGenerator,
)
from app.retrieval.indexer import ChunkIndexer, _chunk_to_metadata
from app.retrieval.vector_store import VectorStore, VectorStoreError, _sanitise_metadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: str = "doc1#c0",
    document_id: str = "doc1",
    text: str = "Sample chunk text.",
    chunk_index: int = 0,
    strategy: str = "fixed",
    source_file: str = "/data/raw/sample.txt",
    file_type: str = "txt",
    page_number: int = 1,
    section_heading: str = "Introduction",
) -> Chunk:
    """Build a Chunk for testing."""
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        chunk_index=chunk_index,
        strategy=strategy,
        source_file=source_file,
        page_number=page_number,
        section_heading=section_heading,
        metadata={"file_type": file_type},
    )


def _fake_embedding(n_dims: int = 4) -> List[float]:
    """Return a fake embedding vector."""
    return [0.1] * n_dims


def _fake_embed_response(texts: List[str]) -> List[List[float]]:
    """Simulate OpenAI v3 Embedding.create returning fake vectors."""
    return [_fake_embedding() for _ in texts]


@pytest.fixture()
def tmp_chroma(tmp_path: Path) -> Path:
    """Return a temporary directory for ChromaDB persistence."""
    return tmp_path / "chroma"


@pytest.fixture()
def vector_store(tmp_chroma: Path) -> VectorStore:
    """Return a VectorStore backed by a temporary ChromaDB directory."""
    return VectorStore(
        persist_directory=str(tmp_chroma),
        collection_name="test_collection",
    )


@pytest.fixture()
def embedding_gen() -> EmbeddingGenerator:
    """Return an EmbeddingGenerator with a fake API key (no real calls)."""
    return EmbeddingGenerator(
        api_key="sk-test-fake",
        model="text-embedding-3-small",
        batch_size=10,
        validate_on_init=True,
    )


@pytest.fixture()
def chunk_indexer(embedding_gen: EmbeddingGenerator, vector_store: VectorStore) -> ChunkIndexer:
    """Return a ChunkIndexer wired with mock embedding gen and tmp vector store."""
    return ChunkIndexer(
        embedding_generator=embedding_gen,
        vector_store=vector_store,
    )


# ---------------------------------------------------------------------------
# EmbeddingGenerator tests
# ---------------------------------------------------------------------------


class TestEmbeddingGeneratorConfig:
    """Config loading and validation tests."""

    def test_explicit_api_key_accepted(self):
        gen = EmbeddingGenerator(api_key="sk-explicit", validate_on_init=True)
        assert gen._api_key == "sk-explicit"

    def test_env_api_key_accepted(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")
        gen = EmbeddingGenerator(validate_on_init=True)
        assert gen._api_key == "sk-env-key"

    def test_missing_api_key_raises_config_error(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # Clear lru_cache so Settings.from_env() is re-evaluated inside the try block
        from app.config import get_settings
        get_settings.cache_clear()
        # Patch the Settings factory directly
        with patch("app.config.Settings.from_env") as mock_from_env:
            mock_settings = MagicMock()
            mock_settings.openai_api_key = None
            mock_from_env.return_value = mock_settings
            with pytest.raises(EmbeddingConfigError, match="OPENAI_API_KEY"):
                EmbeddingGenerator(api_key=None, validate_on_init=True)
        get_settings.cache_clear()

    def test_no_validation_on_init_allows_missing_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from app.config import get_settings
        get_settings.cache_clear()
        with patch("app.config.Settings.from_env") as mock_from_env:
            mock_settings = MagicMock()
            mock_settings.openai_api_key = None
            mock_from_env.return_value = mock_settings
            gen = EmbeddingGenerator(api_key=None, validate_on_init=False)
            assert gen._api_key is None
        get_settings.cache_clear()

    def test_model_defaults_to_text_embedding_3_small(self, monkeypatch):
        monkeypatch.delenv("OPENAI_EMBEDDING_MODEL", raising=False)
        gen = EmbeddingGenerator(api_key="sk-test", validate_on_init=True)
        assert gen.model == "text-embedding-3-small"

    def test_explicit_model_overrides_default(self):
        gen = EmbeddingGenerator(
            api_key="sk-test",
            model="text-embedding-ada-002",
            validate_on_init=True,
        )
        assert gen.model == "text-embedding-ada-002"

    def test_batch_size_clamped_to_min_1(self):
        gen = EmbeddingGenerator(api_key="sk-test", batch_size=0, validate_on_init=True)
        assert gen.batch_size == 1

    def test_env_model_is_used(self, monkeypatch):
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
        gen = EmbeddingGenerator(api_key="sk-test", validate_on_init=True)
        assert gen.model == "text-embedding-ada-002"


class TestEmbeddingGeneratorEmbedTexts:
    """embed_texts batching and error handling."""

    def _mock_call_api(self, texts):
        return _fake_embed_response(texts)

    def test_empty_input_returns_empty_list(self, embedding_gen: EmbeddingGenerator):
        result = embedding_gen.embed_texts([])
        assert result == []

    def test_single_text_returns_one_vector(self, embedding_gen: EmbeddingGenerator):
        with patch.object(embedding_gen, "_call_api", side_effect=self._mock_call_api):
            result = embedding_gen.embed_texts(["hello"])
        assert len(result) == 1
        assert isinstance(result[0], list)

    def test_multiple_texts_returns_matching_count(self, embedding_gen: EmbeddingGenerator):
        texts = ["text one", "text two", "text three"]
        with patch.object(embedding_gen, "_call_api", side_effect=self._mock_call_api):
            result = embedding_gen.embed_texts(texts)
        assert len(result) == len(texts)

    def test_batching_splits_into_correct_number_of_api_calls(self):
        """batch_size=2 on 5 texts should produce 3 API calls."""
        gen = EmbeddingGenerator(api_key="sk-test", batch_size=2, validate_on_init=True)
        api_calls: list = []

        def mock_call(texts):
            api_calls.append(len(texts))
            return _fake_embed_response(texts)

        texts = ["a", "b", "c", "d", "e"]
        with patch.object(gen, "_call_api", side_effect=mock_call):
            result = gen.embed_texts(texts)

        assert result is not None
        assert len(result) == 5
        # Expected: [2, 2, 1]
        assert api_calls == [2, 2, 1]

    def test_api_error_raises_embedding_api_error(self, embedding_gen: EmbeddingGenerator):
        def failing_call(texts):
            raise RuntimeError("API timeout")

        with patch.object(embedding_gen, "_call_api", side_effect=failing_call):
            with pytest.raises(EmbeddingAPIError, match="API timeout"):
                embedding_gen.embed_texts(["some text"])

    def test_missing_key_at_call_time_raises_config_error(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        gen = EmbeddingGenerator(api_key=None, validate_on_init=False)
        with pytest.raises(EmbeddingConfigError):
            gen.embed_texts(["text"])

    def test_embed_query_returns_single_vector(self, embedding_gen: EmbeddingGenerator):
        with patch.object(embedding_gen, "_call_api", side_effect=self._mock_call_api):
            vec = embedding_gen.embed_query("my query")
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)


# ---------------------------------------------------------------------------
# VectorStore tests
# ---------------------------------------------------------------------------


class TestVectorStoreInit:
    """Initialisation and persistence tests."""

    def test_creates_directory_if_missing(self, tmp_path: Path):
        chroma_dir = tmp_path / "new_chroma"
        assert not chroma_dir.exists()
        vs = VectorStore(persist_directory=str(chroma_dir), collection_name="test")
        assert chroma_dir.exists()

    def test_collection_created_on_first_open(self, tmp_chroma: Path):
        vs = VectorStore(persist_directory=str(tmp_chroma), collection_name="col1")
        assert vs.count() == 0

    def test_collection_can_be_reopened(self, tmp_chroma: Path):
        # First open: insert data
        vs1 = VectorStore(persist_directory=str(tmp_chroma), collection_name="col_reopen")
        vs1.upsert_chunks(
            ids=["id1"],
            documents=["doc text"],
            embeddings=[[0.1, 0.2, 0.3]],
            metadatas=[{"key": "value"}],
        )
        assert vs1.count() == 1

        # Second open of same directory/collection
        vs2 = VectorStore(persist_directory=str(tmp_chroma), collection_name="col_reopen")
        assert vs2.count() == 1

    def test_count_returns_integer(self, vector_store: VectorStore):
        assert isinstance(vector_store.count(), int)

    def test_get_collection_info(self, vector_store: VectorStore):
        info = vector_store.get_collection_info()
        assert "collection_name" in info
        assert "persist_directory" in info
        assert "chunk_count" in info
        assert info["chunk_count"] == 0


class TestVectorStoreUpsert:
    """Upsert and deduplication tests."""

    def _sample_data(self, n: int = 2):
        ids = [f"chunk_{i}" for i in range(n)]
        docs = [f"Document text number {i}" for i in range(n)]
        embeddings = [[float(i)] * 3 for i in range(n)]
        metadatas = [{"index": i, "source": "test"} for i in range(n)]
        return ids, docs, embeddings, metadatas

    def test_upsert_increases_count(self, vector_store: VectorStore):
        ids, docs, embs, metas = self._sample_data(3)
        vector_store.upsert_chunks(ids, docs, embs, metas)
        assert vector_store.count() == 3

    def test_upsert_empty_ids_raises_value_error(self, vector_store: VectorStore):
        with pytest.raises(ValueError, match="empty ids"):
            vector_store.upsert_chunks([], [], [], [])

    def test_upsert_mismatched_lengths_raises_value_error(self, vector_store: VectorStore):
        with pytest.raises(ValueError, match="Mismatched lengths"):
            vector_store.upsert_chunks(
                ids=["a"],
                documents=["doc"],
                embeddings=[[0.1], [0.2]],  # extra item
                metadatas=[{}],
            )

    def test_duplicate_ids_do_not_create_duplicates(self, vector_store: VectorStore):
        """Upserting the same IDs twice must not create duplicate records."""
        ids, docs, embs, metas = self._sample_data(2)
        vector_store.upsert_chunks(ids, docs, embs, metas)
        assert vector_store.count() == 2

        # Upsert same IDs again (simulating re-indexing)
        vector_store.upsert_chunks(ids, docs, embs, metas)
        assert vector_store.count() == 2  # still 2, not 4

    def test_metadata_is_preserved(self, vector_store: VectorStore):
        """Metadata values must round-trip through ChromaDB."""
        vector_store.upsert_chunks(
            ids=["meta_chunk"],
            documents=["text with metadata"],
            embeddings=[[0.5, 0.5]],
            metadatas=[{
                "document_id": "doc_abc",
                "chunk_index": 3,
                "source_file": "/data/raw/test.md",
                "chunking_strategy": "fixed",
            }],
        )
        result = vector_store._collection.get(ids=["meta_chunk"])
        stored_meta = result["metadatas"][0]
        assert stored_meta["document_id"] == "doc_abc"
        assert stored_meta["chunk_index"] == 3
        assert stored_meta["chunking_strategy"] == "fixed"

    def test_chunk_exists_false_when_not_indexed(self, vector_store: VectorStore):
        assert vector_store.chunk_exists("nonexistent_id") is False

    def test_chunk_exists_true_after_upsert(self, vector_store: VectorStore):
        vector_store.upsert_chunks(
            ids=["exists_check"],
            documents=["text"],
            embeddings=[[1.0]],
            metadatas=[{"key": "val"}],
        )
        assert vector_store.chunk_exists("exists_check") is True


class TestSanitiseMetadata:
    """Unit tests for the _sanitise_metadata helper."""

    def test_none_becomes_empty_string(self):
        result = _sanitise_metadata({"key": None})
        assert result["key"] == ""

    def test_scalars_pass_through(self):
        meta = {"s": "str", "i": 1, "f": 1.5, "b": True}
        result = _sanitise_metadata(meta)
        assert result == meta

    def test_non_scalar_converted_to_string(self):
        result = _sanitise_metadata({"lst": [1, 2, 3]})
        assert result["lst"] == "[1, 2, 3]"

    def test_empty_dict_returns_empty_dict(self):
        assert _sanitise_metadata({}) == {}


# ---------------------------------------------------------------------------
# ChunkIndexer tests
# ---------------------------------------------------------------------------


class TestChunkIndexer:
    """Integration tests for ChunkIndexer (mocked embeddings, real ChromaDB)."""

    def _mock_embed(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def test_index_single_chunk(
        self, chunk_indexer: ChunkIndexer, vector_store: VectorStore, embedding_gen: EmbeddingGenerator
    ):
        chunk = _make_chunk()
        with patch.object(embedding_gen, "_call_api", side_effect=self._mock_embed):
            indexed = chunk_indexer.index_chunks([chunk])
        assert indexed == 1
        assert vector_store.count() == 1

    def test_index_multiple_chunks(
        self, chunk_indexer: ChunkIndexer, vector_store: VectorStore, embedding_gen: EmbeddingGenerator
    ):
        chunks = [
            _make_chunk(chunk_id=f"doc1#c{i}", chunk_index=i, text=f"Chunk text {i}")
            for i in range(5)
        ]
        with patch.object(embedding_gen, "_call_api", side_effect=self._mock_embed):
            indexed = chunk_indexer.index_chunks(chunks)
        assert indexed == 5
        assert vector_store.count() == 5

    def test_index_empty_list_returns_zero(self, chunk_indexer: ChunkIndexer):
        indexed = chunk_indexer.index_chunks([])
        assert indexed == 0

    def test_re_indexing_same_chunks_does_not_duplicate(
        self, chunk_indexer: ChunkIndexer, vector_store: VectorStore, embedding_gen: EmbeddingGenerator
    ):
        chunks = [_make_chunk(chunk_id="stable_id", chunk_index=0)]
        with patch.object(embedding_gen, "_call_api", side_effect=self._mock_embed):
            chunk_indexer.index_chunks(chunks)
            chunk_indexer.index_chunks(chunks)  # second pass
        assert vector_store.count() == 1  # idempotent

    def test_deterministic_ids_are_stable(
        self, chunk_indexer: ChunkIndexer, embedding_gen: EmbeddingGenerator
    ):
        chunk = _make_chunk(chunk_id="doc1#c0", chunk_index=0)
        with patch.object(embedding_gen, "_call_api", side_effect=self._mock_embed):
            chunk_indexer.index_chunks([chunk])
        assert chunk_indexer.vector_store.chunk_exists("doc1#c0")

    def test_embedding_api_failure_skips_batch(
        self, chunk_indexer: ChunkIndexer, vector_store: VectorStore, embedding_gen: EmbeddingGenerator
    ):
        def failing_call(texts):
            raise RuntimeError("API down")

        chunk = _make_chunk()
        with patch.object(embedding_gen, "_call_api", side_effect=failing_call):
            indexed = chunk_indexer.index_chunks([chunk])
        assert indexed == 0
        assert vector_store.count() == 0

    def test_metadata_stored_per_chunk(
        self, chunk_indexer: ChunkIndexer, vector_store: VectorStore, embedding_gen: EmbeddingGenerator
    ):
        chunk = _make_chunk(
            chunk_id="meta_test_chunk",
            document_id="doc_meta",
            source_file="/data/raw/report.pdf",
            file_type="pdf",
            page_number=5,
            section_heading="Results",
        )
        with patch.object(embedding_gen, "_call_api", side_effect=self._mock_embed):
            chunk_indexer.index_chunks([chunk])
        result = vector_store._collection.get(ids=["meta_test_chunk"])
        meta = result["metadatas"][0]
        assert meta["document_id"] == "doc_meta"
        assert meta["source_file"] == "/data/raw/report.pdf"
        assert meta["page_number"] == 5
        assert meta["section_heading"] == "Results"
        assert meta["chunking_strategy"] == "fixed"


class TestChunkToMetadata:
    """Unit tests for the _chunk_to_metadata helper."""

    def test_all_required_fields_present(self):
        chunk = _make_chunk()
        meta = _chunk_to_metadata(chunk)
        assert "document_id" in meta
        assert "chunk_id" in meta
        assert "source_file" in meta
        assert "chunk_index" in meta
        assert "character_count" in meta
        assert "chunking_strategy" in meta

    def test_page_number_included_when_set(self):
        chunk = _make_chunk(page_number=3)
        meta = _chunk_to_metadata(chunk)
        assert meta["page_number"] == 3

    def test_section_heading_included_when_set(self):
        chunk = _make_chunk(section_heading="Methods")
        meta = _chunk_to_metadata(chunk)
        assert meta["section_heading"] == "Methods"

    def test_file_type_from_chunk_metadata(self):
        chunk = _make_chunk(file_type="pdf")
        meta = _chunk_to_metadata(chunk)
        assert meta.get("file_type") == "pdf"


# ---------------------------------------------------------------------------
# CLI helper: load_chunks_from_directory
# ---------------------------------------------------------------------------


class TestLoadChunksFromDirectory:
    """Tests for the scripts/index.py load helper."""

    def test_loads_valid_chunk_files(self, tmp_path: Path):
        from scripts.index import load_chunks_from_directory

        chunk = _make_chunk()
        chunk_file = tmp_path / f"{chunk.chunk_id.replace('#', '__')}.json"
        chunk_file.write_text(
            json.dumps(chunk.model_dump(), ensure_ascii=False), encoding="utf-8"
        )

        chunks = load_chunks_from_directory(tmp_path)
        assert len(chunks) == 1
        assert chunks[0].chunk_id == chunk.chunk_id

    def test_empty_directory_returns_empty_list(self, tmp_path: Path):
        from scripts.index import load_chunks_from_directory

        chunks = load_chunks_from_directory(tmp_path)
        assert chunks == []

    def test_invalid_json_file_skipped(self, tmp_path: Path):
        from scripts.index import load_chunks_from_directory

        bad_file = tmp_path / "bad_chunk.json"
        bad_file.write_text("{ not valid json }", encoding="utf-8")

        # Also write one good chunk
        chunk = _make_chunk(chunk_id="doc1#c1", chunk_index=1)
        good_file = tmp_path / "doc1__c1.json"
        good_file.write_text(json.dumps(chunk.model_dump()), encoding="utf-8")

        chunks = load_chunks_from_directory(tmp_path)
        assert len(chunks) == 1

    def test_missing_required_field_skipped(self, tmp_path: Path):
        from scripts.index import load_chunks_from_directory

        # Write JSON missing required 'text' field
        incomplete = tmp_path / "incomplete.json"
        incomplete.write_text(json.dumps({"chunk_id": "x", "document_id": "y"}), encoding="utf-8")

        chunks = load_chunks_from_directory(tmp_path)
        assert chunks == []
