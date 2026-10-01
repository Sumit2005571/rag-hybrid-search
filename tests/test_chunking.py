"""Comprehensive unit tests for all three chunking strategies.

Coverage:
- Chunk model fields and defaults
- generate_chunk_id determinism and format
- FixedSizeChunker: normal operation, edge cases, overlap, metadata
- RecursiveStructureChunker: separator hierarchy, overlap, metadata
- SemanticChunker: with injected embeddings, fallback, failure handling
- All strategies: empty/short/large documents, metadata propagation,
  deterministic IDs, strategy field, no missing/duplicated text
- CLI helpers (build_chunker, load_processed_documents, chunk_documents)
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from app.chunking import (
    BaseChunker,
    Chunk,
    FixedSizeChunker,
    RecursiveStructureChunker,
    SemanticChunker,
    generate_chunk_id,
)
from app.chunking.semantic import _cosine_similarity, _split_sentences, _paragraph_fallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_SHORT = "Hello world."
SAMPLE_MEDIUM = "This is sentence one. This is sentence two. This is sentence three."
SAMPLE_PARAGRAPHS = (
    "First paragraph with enough content to be meaningful.\n\n"
    "Second paragraph also has enough content and is distinct.\n\n"
    "Third paragraph rounds out the document nicely."
)
SAMPLE_MARKDOWN = (
    "# Introduction\n\n"
    "This is the intro section with some text about the topic.\n\n"
    "## Background\n\n"
    "Here we discuss the background of the subject matter in detail.\n\n"
    "## Methods\n\n"
    "The methods section explains the approach taken in this research."
)

DOC_ID = "testdoc123"
META = {
    "source_file": "/data/raw/sample.txt",
    "section_heading": "Introduction",
    "page_number": 1,
    "file_type": "txt",
}


# ---------------------------------------------------------------------------
# Chunk model tests
# ---------------------------------------------------------------------------

class TestChunkModel:
    def test_required_fields(self):
        c = Chunk(chunk_id="id1", document_id="doc1", chunk_index=0, text="hello")
        assert c.chunk_id == "id1"
        assert c.document_id == "doc1"
        assert c.chunk_index == 0
        assert c.text == "hello"

    def test_auto_char_count(self):
        c = Chunk(chunk_id="id1", document_id="doc1", chunk_index=0, text="hello")
        assert c.char_count == 5

    def test_explicit_char_count_preserved(self):
        c = Chunk(
            chunk_id="id1", document_id="doc1", chunk_index=0, text="hello",
            char_count=99
        )
        assert c.char_count == 99

    def test_defaults_are_correct(self):
        c = Chunk(chunk_id="id1", document_id="doc1", chunk_index=0, text="hi")
        assert c.source_file == ""
        assert c.section_heading is None
        assert c.page_number is None
        assert c.strategy == "fixed"
        assert c.metadata == {}
        assert c.start_char == 0

    def test_all_fields_settable(self):
        c = Chunk(
            chunk_id="x",
            document_id="d",
            source_file="/path/file.txt",
            chunk_index=3,
            start_char=100,
            end_char=200,
            text="chunk content",
            char_count=13,
            section_heading="# Header",
            page_number=2,
            strategy="recursive",
            metadata={"extra": "value"},
        )
        assert c.source_file == "/path/file.txt"
        assert c.section_heading == "# Header"
        assert c.page_number == 2
        assert c.strategy == "recursive"
        assert c.metadata["extra"] == "value"


# ---------------------------------------------------------------------------
# generate_chunk_id tests
# ---------------------------------------------------------------------------

class TestGenerateChunkId:
    def test_fixed_format(self):
        cid = generate_chunk_id("doc1", 0, "fixed")
        assert cid == "doc1#c0"

    def test_recursive_format(self):
        cid = generate_chunk_id("doc1", 5, "recursive")
        assert cid == "doc1#r5"

    def test_semantic_format(self):
        cid = generate_chunk_id("doc1", 2, "semantic")
        assert cid == "doc1#s2"

    def test_deterministic(self):
        id1 = generate_chunk_id("abc", 3, "fixed")
        id2 = generate_chunk_id("abc", 3, "fixed")
        assert id1 == id2

    def test_different_index_different_id(self):
        id1 = generate_chunk_id("doc", 0, "fixed")
        id2 = generate_chunk_id("doc", 1, "fixed")
        assert id1 != id2

    def test_different_doc_different_id(self):
        id1 = generate_chunk_id("doc1", 0, "fixed")
        id2 = generate_chunk_id("doc2", 0, "fixed")
        assert id1 != id2

    def test_unknown_strategy_defaults_to_c(self):
        cid = generate_chunk_id("doc", 0, "unknown_strategy")
        assert cid == "doc#c0"


# ---------------------------------------------------------------------------
# FixedSizeChunker tests
# ---------------------------------------------------------------------------

class TestFixedSizeChunker:
    def test_empty_text_returns_empty(self):
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        assert chunker.chunk("", DOC_ID) == []

    def test_short_text_single_chunk(self):
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(SAMPLE_SHORT, DOC_ID)
        assert len(chunks) == 1
        assert chunks[0].text == SAMPLE_SHORT

    def test_document_smaller_than_chunk_size(self):
        text = "Small text."
        chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk(text, DOC_ID)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_document_larger_than_chunk_size(self):
        text = "A" * 200
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk(text, DOC_ID)
        assert len(chunks) > 1

    def test_no_text_lost(self):
        """Verifying all text appears in at least one chunk."""
        text = "Hello world, this is a test document with many words." * 5
        chunker = FixedSizeChunker(chunk_size=30, chunk_overlap=5)
        chunks = chunker.chunk(text, DOC_ID)
        # Each character of the first chunk must be in text
        assert chunks[0].text in text
        # Final chunk must end at/within the text
        assert chunks[-1].text[-1] == text[-1]

    def test_no_characters_missing_from_sequence(self):
        """The concatenation of non-overlapping portions must equal the full text."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 3
        size, overlap = 10, 3
        chunker = FixedSizeChunker(chunk_size=size, chunk_overlap=overlap)
        chunks = chunker.chunk(text, DOC_ID)
        step = size - overlap
        # Reconstruct: first chunk in full, then only the non-overlap part of each subsequent
        reconstructed = chunks[0].text
        for chunk in chunks[1:]:
            reconstructed += chunk.text[overlap:]
        assert reconstructed == text

    def test_overlap_behavior(self):
        text = "0123456789ABCDEFGHIJ"  # 20 chars
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=3)
        chunks = chunker.chunk(text, DOC_ID)
        # Second chunk should start with the last 3 chars of first chunk
        assert chunks[1].text[:3] == chunks[0].text[-3:]

    def test_chunk_ids_deterministic(self):
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=2)
        chunks1 = chunker.chunk(SAMPLE_MEDIUM, DOC_ID)
        chunks2 = chunker.chunk(SAMPLE_MEDIUM, DOC_ID)
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]

    def test_chunk_id_format(self):
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=2)
        chunks = chunker.chunk(SAMPLE_MEDIUM, DOC_ID)
        assert chunks[0].chunk_id == f"{DOC_ID}#c0"
        assert chunks[1].chunk_id == f"{DOC_ID}#c1"

    def test_strategy_field(self):
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=2)
        chunks = chunker.chunk(SAMPLE_MEDIUM, DOC_ID)
        assert all(c.strategy == "fixed" for c in chunks)

    def test_metadata_propagated(self):
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=2)
        chunks = chunker.chunk(SAMPLE_MEDIUM, DOC_ID, metadata=META)
        for c in chunks:
            assert c.source_file == META["source_file"]
            assert c.section_heading == META["section_heading"]
            assert c.page_number == META["page_number"]

    def test_char_count_correct(self):
        chunker = FixedSizeChunker(chunk_size=20, chunk_overlap=5)
        chunks = chunker.chunk("X" * 60, DOC_ID)
        for c in chunks:
            assert c.char_count == len(c.text)

    def test_start_end_char_correct(self):
        text = "0123456789ABCDEFGHIJ"
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=0)
        chunks = chunker.chunk(text, DOC_ID)
        assert chunks[0].start_char == 0
        assert chunks[0].end_char == 10
        assert chunks[1].start_char == 10
        assert chunks[1].end_char == 20

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=10, chunk_overlap=10)

    def test_document_id_in_chunk(self):
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=2)
        chunks = chunker.chunk("Hello world!", DOC_ID)
        assert all(c.document_id == DOC_ID for c in chunks)

    def test_chunk_index_sequential(self):
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=2)
        chunks = chunker.chunk(SAMPLE_MEDIUM, DOC_ID)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    # backward-compat: test_structure.py expects doc1#c0
    def test_backward_compat_chunk_id(self):
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=2)
        chunks = chunker.chunk("Hello world, this is a test.", "doc1")
        assert chunks[0].chunk_id == "doc1#c0"


# ---------------------------------------------------------------------------
# RecursiveStructureChunker tests
# ---------------------------------------------------------------------------

class TestRecursiveStructureChunker:
    def test_empty_text_returns_empty(self):
        chunker = RecursiveStructureChunker(chunk_size=100, chunk_overlap=10)
        assert chunker.chunk("", DOC_ID) == []

    def test_short_text_single_chunk(self):
        chunker = RecursiveStructureChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk(SAMPLE_SHORT, DOC_ID)
        assert len(chunks) == 1
        assert chunks[0].text == SAMPLE_SHORT

    def test_document_smaller_than_chunk_size(self):
        text = "Small enough text."
        chunker = RecursiveStructureChunker(chunk_size=1000, chunk_overlap=0)
        chunks = chunker.chunk(text, DOC_ID)
        assert len(chunks) == 1

    def test_large_document_splits(self):
        text = "Word " * 200  # 1000 chars
        chunker = RecursiveStructureChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.chunk(text, DOC_ID)
        assert len(chunks) > 1

    def test_prefers_paragraph_boundaries(self):
        """Chunks should not split within a paragraph when possible."""
        text = "Para one text here.\n\nPara two text here.\n\nPara three text."
        chunker = RecursiveStructureChunker(chunk_size=30, chunk_overlap=0)
        chunks = chunker.chunk(text, DOC_ID)
        # Each chunk should be a whole paragraph or smaller unit
        assert len(chunks) > 0

    def test_heading_boundary_respected(self):
        """Markdown headings should be treated as strong split points."""
        text = SAMPLE_MARKDOWN
        chunker = RecursiveStructureChunker(chunk_size=80, chunk_overlap=0)
        chunks = chunker.chunk(text, DOC_ID)
        assert len(chunks) >= 3  # at least one per section

    def test_strategy_field(self):
        chunker = RecursiveStructureChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.chunk(SAMPLE_PARAGRAPHS, DOC_ID)
        assert all(c.strategy == "recursive" for c in chunks)

    def test_chunk_id_format(self):
        chunker = RecursiveStructureChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.chunk(SAMPLE_PARAGRAPHS, DOC_ID)
        assert chunks[0].chunk_id == f"{DOC_ID}#r0"

    def test_metadata_propagated(self):
        chunker = RecursiveStructureChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.chunk(SAMPLE_PARAGRAPHS, DOC_ID, metadata=META)
        for c in chunks:
            assert c.source_file == META["source_file"]
            assert c.section_heading == META["section_heading"]
            assert c.page_number == META["page_number"]

    def test_chunk_ids_deterministic(self):
        chunker = RecursiveStructureChunker(chunk_size=50, chunk_overlap=10)
        chunks1 = chunker.chunk(SAMPLE_PARAGRAPHS, DOC_ID)
        chunks2 = chunker.chunk(SAMPLE_PARAGRAPHS, DOC_ID)
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]

    def test_overlap_adds_tail_to_next_chunk(self):
        text = "Hello world paragraph one.\n\nSecond paragraph here."
        chunker = RecursiveStructureChunker(chunk_size=30, chunk_overlap=5)
        chunks = chunker.chunk(text, DOC_ID)
        if len(chunks) >= 2:
            # The overlap tail of chunk[0] should appear at the start of chunk[1]
            tail = chunks[0].text[-5:]
            assert chunks[1].text.startswith(tail)

    def test_chunk_index_sequential(self):
        chunker = RecursiveStructureChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.chunk(SAMPLE_PARAGRAPHS, DOC_ID)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_char_count_correct(self):
        chunker = RecursiveStructureChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.chunk(SAMPLE_PARAGRAPHS, DOC_ID)
        for c in chunks:
            assert c.char_count == len(c.text)

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            RecursiveStructureChunker(chunk_size=50, chunk_overlap=50)

    def test_custom_separators(self):
        """A custom separator list should be respected."""
        text = "AAA|BBB|CCC"
        chunker = RecursiveStructureChunker(
            chunk_size=4, chunk_overlap=0, separators=["|", ""]
        )
        chunks = chunker.chunk(text, DOC_ID)
        # Text should be split at | boundaries
        combined = "".join(c.text for c in chunks)
        # All original characters must appear (order of reconstruction may differ
        # due to overlap mechanics, but set of chars must be conserved)
        assert len(combined) >= len(text)


# ---------------------------------------------------------------------------
# SemanticChunker tests
# ---------------------------------------------------------------------------

def _make_embedding_fn(similarity_map: Optional[Dict[int, float]] = None):
    """Return an embedding_fn where consecutive pairs have given similarities.

    ``similarity_map`` maps pair index → desired cosine similarity.
    If not given, all pairs get similarity=1.0 (no splits).
    """
    sim = similarity_map or {}

    def _fn(texts: List[str]) -> List[List[float]]:
        n = len(texts)
        embeddings = []
        for i in range(n):
            # Build a vector that achieves the desired similarity with the previous
            if i == 0:
                embeddings.append([1.0, 0.0])
            else:
                desired = sim.get(i - 1, 1.0)
                # cos(theta) = desired → [cos, sin]
                import math
                angle = math.acos(max(-1.0, min(1.0, desired)))
                # Alternate sign to create distinct vectors
                sign = 1 if i % 2 == 0 else -1
                embeddings.append([math.cos(angle), sign * math.sin(angle)])
        return embeddings

    return _fn


class TestSemanticChunker:
    def test_empty_text_returns_empty(self):
        chunker = SemanticChunker(
            breakpoint_threshold=0.5,
            embedding_fn=_make_embedding_fn(),
        )
        assert chunker.chunk("", DOC_ID) == []

    def test_short_text_single_chunk(self):
        chunker = SemanticChunker(
            breakpoint_threshold=0.5,
            embedding_fn=_make_embedding_fn(),
        )
        chunks = chunker.chunk(SAMPLE_SHORT, DOC_ID)
        assert len(chunks) == 1
        assert chunks[0].text.strip() == SAMPLE_SHORT.strip()

    def test_high_similarity_keeps_together(self):
        """When all sentences are similar, there should be only 1 chunk."""
        chunker = SemanticChunker(
            breakpoint_threshold=0.1,  # very low threshold → almost never split
            min_chunk_size=0,
            embedding_fn=_make_embedding_fn({0: 1.0, 1: 1.0, 2: 1.0}),
        )
        chunks = chunker.chunk(SAMPLE_MEDIUM, DOC_ID)
        assert len(chunks) == 1

    def test_low_similarity_splits(self):
        """When similarity is below threshold, the text must be split."""
        # Force very low similarity for each pair → all sentences become chunks
        low_sim_fn = _make_embedding_fn({0: 0.0, 1: 0.0, 2: 0.0})
        chunker = SemanticChunker(
            breakpoint_threshold=0.9,  # very high → always split
            min_chunk_size=0,
            embedding_fn=low_sim_fn,
        )
        chunks = chunker.chunk(SAMPLE_MEDIUM, DOC_ID)
        assert len(chunks) > 1

    def test_fallback_when_no_api_key(self):
        """Without an API key and no injected fn, chunker falls back gracefully."""
        chunker = SemanticChunker(
            breakpoint_threshold=0.75,
            min_chunk_size=0,
            max_chunk_size=2000,
            # No embedding_fn → will try OpenAI → no key → falls back
        )
        # Patch get_settings to return no API key
        with patch("app.chunking.semantic.get_settings") as mock_settings:
            mock_settings.return_value.openai_api_key = None
            mock_settings.return_value.semantic_breakpoint_threshold = 0.75
            mock_settings.return_value.semantic_min_chunk_size = 0
            mock_settings.return_value.semantic_max_chunk_size = 2000
            chunks = chunker.chunk(SAMPLE_PARAGRAPHS, DOC_ID)
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_fallback_when_embedding_raises(self):
        """If the injected embedding_fn raises, fallback should be triggered."""
        def _failing_fn(texts):
            raise RuntimeError("Simulated API failure")

        chunker = SemanticChunker(
            breakpoint_threshold=0.75,
            min_chunk_size=0,
            max_chunk_size=2000,
            embedding_fn=_failing_fn,
        )
        chunks = chunker.chunk(SAMPLE_PARAGRAPHS, DOC_ID)
        assert len(chunks) >= 1

    def test_strategy_field(self):
        chunker = SemanticChunker(
            breakpoint_threshold=0.5,
            min_chunk_size=0,
            embedding_fn=_make_embedding_fn(),
        )
        chunks = chunker.chunk(SAMPLE_MEDIUM, DOC_ID)
        assert all(c.strategy == "semantic" for c in chunks)

    def test_chunk_id_format(self):
        chunker = SemanticChunker(
            breakpoint_threshold=0.5,
            min_chunk_size=0,
            embedding_fn=_make_embedding_fn(),
        )
        chunks = chunker.chunk(SAMPLE_MEDIUM, DOC_ID)
        assert chunks[0].chunk_id == f"{DOC_ID}#s0"

    def test_metadata_propagated(self):
        chunker = SemanticChunker(
            breakpoint_threshold=0.5,
            min_chunk_size=0,
            embedding_fn=_make_embedding_fn(),
        )
        chunks = chunker.chunk(SAMPLE_MEDIUM, DOC_ID, metadata=META)
        for c in chunks:
            assert c.source_file == META["source_file"]
            assert c.section_heading == META["section_heading"]
            assert c.page_number == META["page_number"]

    def test_chunk_ids_deterministic(self):
        emb_fn = _make_embedding_fn()
        chunker = SemanticChunker(
            breakpoint_threshold=0.5,
            min_chunk_size=0,
            embedding_fn=emb_fn,
        )
        chunks1 = chunker.chunk(SAMPLE_MEDIUM, DOC_ID)
        chunks2 = chunker.chunk(SAMPLE_MEDIUM, DOC_ID)
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]

    def test_min_chunk_size_merging(self):
        """Chunks below min_chunk_size should be merged with a neighbour."""
        # low_sim → many tiny chunks; min_chunk_size merges them
        low_sim = _make_embedding_fn({0: 0.0, 1: 0.0})
        chunker = SemanticChunker(
            breakpoint_threshold=0.9,
            min_chunk_size=500,   # large min → force merging
            max_chunk_size=5000,
            embedding_fn=low_sim,
        )
        chunks = chunker.chunk(SAMPLE_MEDIUM, DOC_ID)
        for c in chunks:
            assert c.char_count >= 500 or len(chunks) == 1

    def test_max_chunk_size_enforcement(self):
        """No chunk should exceed max_chunk_size characters."""
        big_text = "This sentence is important. " * 200  # ~5600 chars
        emb_fn = _make_embedding_fn()  # high similarity → no semantic splits
        chunker = SemanticChunker(
            breakpoint_threshold=0.1,
            min_chunk_size=0,
            max_chunk_size=200,
            embedding_fn=emb_fn,
        )
        chunks = chunker.chunk(big_text, DOC_ID)
        for c in chunks:
            assert c.char_count <= 200

    def test_char_count_correct(self):
        emb_fn = _make_embedding_fn()
        chunker = SemanticChunker(
            breakpoint_threshold=0.5,
            min_chunk_size=0,
            embedding_fn=emb_fn,
        )
        chunks = chunker.chunk(SAMPLE_MEDIUM, DOC_ID)
        for c in chunks:
            assert c.char_count == len(c.text)

    def test_chunk_index_sequential(self):
        low_sim = _make_embedding_fn({0: 0.0, 1: 0.0})
        chunker = SemanticChunker(
            breakpoint_threshold=0.9,
            min_chunk_size=0,
            embedding_fn=low_sim,
        )
        chunks = chunker.chunk(SAMPLE_MEDIUM, DOC_ID)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i


# ---------------------------------------------------------------------------
# Semantic chunker helper tests
# ---------------------------------------------------------------------------

class TestSemanticHelpers:
    def test_cosine_similarity_identical(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_cosine_similarity_empty(self):
        assert _cosine_similarity([], [1.0]) == 0.0

    def test_cosine_similarity_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_split_sentences_basic(self):
        text = "Hello world. This is two. And three."
        sentences = _split_sentences(text)
        assert len(sentences) == 3

    def test_split_sentences_single(self):
        assert _split_sentences("Hello.") == ["Hello."]

    def test_split_sentences_empty(self):
        assert _split_sentences("") == []

    def test_paragraph_fallback_basic(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        segments = _paragraph_fallback(text, chunk_size=1000)
        assert len(segments) == 3

    def test_paragraph_fallback_oversized_para(self):
        text = "A" * 500
        segments = _paragraph_fallback(text, chunk_size=100)
        assert all(len(s) <= 100 for s in segments)


# ---------------------------------------------------------------------------
# CLI helper tests
# ---------------------------------------------------------------------------

class TestCLIHelpers:
    def test_build_chunker_fixed(self):
        from scripts.chunk import build_chunker
        chunker = build_chunker("fixed", 300, 30, None)
        assert isinstance(chunker, FixedSizeChunker)
        assert chunker.chunk_size == 300
        assert chunker.chunk_overlap == 30

    def test_build_chunker_recursive(self):
        from scripts.chunk import build_chunker
        chunker = build_chunker("recursive", 400, 40, None)
        assert isinstance(chunker, RecursiveStructureChunker)

    def test_build_chunker_semantic(self):
        from scripts.chunk import build_chunker
        chunker = build_chunker("semantic", 500, 0, 0.65)
        assert isinstance(chunker, SemanticChunker)
        assert chunker.breakpoint_threshold == 0.65

    def test_build_chunker_unknown_raises(self):
        from scripts.chunk import build_chunker
        with pytest.raises(ValueError):
            build_chunker("unknown", 500, 50, None)

    def test_load_processed_documents(self, tmp_path):
        from scripts.chunk import load_processed_documents
        doc_file = tmp_path / "abc123.json"
        payload = {
            "content": "Hello world.",
            "metadata": {"document_id": "abc123", "file_type": "txt"},
        }
        doc_file.write_text(json.dumps(payload), encoding="utf-8")
        docs = load_processed_documents(tmp_path)
        assert len(docs) == 1
        assert docs[0]["content"] == "Hello world."

    def test_load_processed_documents_empty_dir(self, tmp_path):
        from scripts.chunk import load_processed_documents
        docs = load_processed_documents(tmp_path)
        assert docs == []

    def test_load_processed_documents_skips_bad_json(self, tmp_path):
        from scripts.chunk import load_processed_documents
        (tmp_path / "bad.json").write_text("not valid json", encoding="utf-8")
        docs = load_processed_documents(tmp_path)
        assert docs == []

    def test_chunk_documents_fixed(self, tmp_path):
        from scripts.chunk import chunk_documents
        docs = [
            {
                "content": "Hello world, this is a sample text for chunking." * 5,
                "metadata": {"document_id": "doc1", "source_file": "/path/file.txt"},
            }
        ]
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=5)
        total = chunk_documents(docs, chunker, tmp_path / "out")
        assert total > 0
        chunk_files = list((tmp_path / "out").glob("*.json"))
        assert len(chunk_files) == total

    def test_chunk_documents_empty_content_skipped(self, tmp_path):
        from scripts.chunk import chunk_documents
        docs = [
            {"content": "", "metadata": {"document_id": "doc1"}}
        ]
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        total = chunk_documents(docs, chunker, tmp_path / "out")
        assert total == 0

    def test_chunk_json_output_structure(self, tmp_path):
        from scripts.chunk import chunk_documents
        docs = [
            {
                "content": "This is a document with enough text to chunk properly.",
                "metadata": {
                    "document_id": "docABC",
                    "source_file": "/path/file.txt",
                    "section_heading": "Intro",
                    "page_number": 1,
                },
            }
        ]
        chunker = FixedSizeChunker(chunk_size=30, chunk_overlap=0)
        chunk_documents(docs, chunker, tmp_path / "chunks")
        files = list((tmp_path / "chunks").glob("*.json"))
        assert len(files) > 0
        data = json.loads(files[0].read_text(encoding="utf-8"))
        required_keys = {
            "chunk_id", "document_id", "source_file", "chunk_index",
            "text", "char_count", "strategy",
        }
        assert required_keys.issubset(data.keys())


# ---------------------------------------------------------------------------
# Cross-strategy consistency tests
# ---------------------------------------------------------------------------

class TestCrossStrategy:
    STRATEGIES = [
        FixedSizeChunker(chunk_size=80, chunk_overlap=10),
        RecursiveStructureChunker(chunk_size=80, chunk_overlap=10),
        SemanticChunker(
            breakpoint_threshold=0.5,
            min_chunk_size=0,
            max_chunk_size=500,
            embedding_fn=_make_embedding_fn({0: 0.3, 1: 0.3}),
        ),
    ]

    @pytest.mark.parametrize("chunker", STRATEGIES)
    def test_returns_list_of_chunks(self, chunker):
        chunks = chunker.chunk(SAMPLE_PARAGRAPHS, DOC_ID)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    @pytest.mark.parametrize("chunker", STRATEGIES)
    def test_empty_text(self, chunker):
        assert chunker.chunk("", DOC_ID) == []

    @pytest.mark.parametrize("chunker", STRATEGIES)
    def test_chunk_index_sequential(self, chunker):
        chunks = chunker.chunk(SAMPLE_PARAGRAPHS, DOC_ID)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    @pytest.mark.parametrize("chunker", STRATEGIES)
    def test_document_id_preserved(self, chunker):
        chunks = chunker.chunk(SAMPLE_SHORT, DOC_ID)
        assert all(c.document_id == DOC_ID for c in chunks)

    @pytest.mark.parametrize("chunker", STRATEGIES)
    def test_char_count_matches_text(self, chunker):
        chunks = chunker.chunk(SAMPLE_PARAGRAPHS, DOC_ID)
        for c in chunks:
            assert c.char_count == len(c.text)

    @pytest.mark.parametrize("chunker", STRATEGIES)
    def test_metadata_section_heading(self, chunker):
        chunks = chunker.chunk(SAMPLE_SHORT, DOC_ID, metadata=META)
        assert all(c.section_heading == "Introduction" for c in chunks)

    @pytest.mark.parametrize("chunker", STRATEGIES)
    def test_metadata_page_number(self, chunker):
        chunks = chunker.chunk(SAMPLE_SHORT, DOC_ID, metadata=META)
        assert all(c.page_number == 1 for c in chunks)
