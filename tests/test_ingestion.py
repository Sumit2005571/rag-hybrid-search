"""Comprehensive unit tests for multi-format document ingestion.

Tests cover:
- TXT loading and normalization
- Markdown heading detection and section splitting
- HTML noise removal and heading preservation
- PDF page-by-page extraction
- Deterministic document IDs (idempotent re-indexing)
- DocumentLoader dispatch and extension registry
- Graceful handling of malformed / unsupported files
- CLI ingest_directory function
"""

import json
import os
from pathlib import Path

import pytest

from app.ingestion.metadata import DocumentMetadata, RawDocument, generate_document_id
from app.ingestion.normalizer import DocumentNormalizer
from app.ingestion.loaders import (
    BaseLoader,
    DocumentLoader,
    HTMLLoader,
    MarkdownLoader,
    PDFLoader,
    TextLoader,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Metadata & deterministic ID tests
# ---------------------------------------------------------------------------

class TestDocumentMetadata:
    def test_metadata_required_fields(self):
        meta = DocumentMetadata(
            source_file="/path/to/file.txt",
            document_id="abc123",
            file_type="txt",
        )
        assert meta.source_file == "/path/to/file.txt"
        assert meta.document_id == "abc123"
        assert meta.file_type == "txt"
        assert meta.ingestion_timestamp  # auto-populated

    def test_metadata_source_alias(self):
        meta = DocumentMetadata(
            source_file="/path/to/file.txt",
            document_id="abc123",
            file_type="txt",
        )
        assert meta.source == "/path/to/file.txt"

    def test_metadata_optional_fields_default_none(self):
        meta = DocumentMetadata(
            source_file="x.txt",
            document_id="id1",
            file_type="txt",
        )
        assert meta.page_number is None
        assert meta.section_heading is None
        assert meta.author is None
        assert meta.extra == {}


class TestDeterministicId:
    def test_same_path_same_id(self):
        id1 = generate_document_id("/path/to/doc.pdf")
        id2 = generate_document_id("/path/to/doc.pdf")
        assert id1 == id2

    def test_different_paths_different_id(self):
        id1 = generate_document_id("/a.txt")
        id2 = generate_document_id("/b.txt")
        assert id1 != id2

    def test_page_number_differentiates(self):
        id1 = generate_document_id("/doc.pdf", page_number=1)
        id2 = generate_document_id("/doc.pdf", page_number=2)
        assert id1 != id2

    def test_id_length(self):
        doc_id = generate_document_id("/some/file.txt")
        assert len(doc_id) == 16


# ---------------------------------------------------------------------------
# TXT loader tests
# ---------------------------------------------------------------------------

class TestTextLoader:
    def test_load_txt(self):
        docs = TextLoader().load(FIXTURES_DIR / "sample.txt")
        assert len(docs) == 1
        doc = docs[0]
        assert "plain text document" in doc.content
        assert doc.metadata.file_type == "txt"
        assert doc.metadata.title == "sample.txt"
        assert doc.metadata.source_file.endswith("sample.txt")

    def test_load_txt_metadata_has_timestamp(self):
        docs = TextLoader().load(FIXTURES_DIR / "sample.txt")
        assert docs[0].metadata.ingestion_timestamp is not None


# ---------------------------------------------------------------------------
# Markdown loader tests
# ---------------------------------------------------------------------------

class TestMarkdownLoader:
    def test_load_md_splits_by_heading(self):
        docs = MarkdownLoader().load(FIXTURES_DIR / "sample.md")
        # sample.md has headings: # Sample Markdown Document, ## Section One,
        # ### Subsection 1.1, ## Section Two → should produce 4 sections
        assert len(docs) >= 4

    def test_md_section_headings_preserved(self):
        docs = MarkdownLoader().load(FIXTURES_DIR / "sample.md")
        headings = [d.metadata.section_heading for d in docs if d.metadata.section_heading]
        assert any("Section One" in h for h in headings)
        assert any("Section Two" in h for h in headings)

    def test_md_file_type(self):
        docs = MarkdownLoader().load(FIXTURES_DIR / "sample.md")
        for doc in docs:
            assert doc.metadata.file_type == "md"

    def test_md_without_headings(self, tmp_path):
        p = tmp_path / "plain.md"
        p.write_text("No headings here.\n\nJust paragraphs.", encoding="utf-8")
        docs = MarkdownLoader().load(p)
        assert len(docs) == 1
        assert docs[0].metadata.section_heading is None

    def test_md_deterministic_ids(self):
        docs1 = MarkdownLoader().load(FIXTURES_DIR / "sample.md")
        docs2 = MarkdownLoader().load(FIXTURES_DIR / "sample.md")
        ids1 = [d.metadata.document_id for d in docs1]
        ids2 = [d.metadata.document_id for d in docs2]
        assert ids1 == ids2


# ---------------------------------------------------------------------------
# HTML loader tests
# ---------------------------------------------------------------------------

class TestHTMLLoader:
    def test_load_html(self):
        docs = HTMLLoader().load(FIXTURES_DIR / "sample.html")
        assert len(docs) == 1

    def test_html_noise_removed(self):
        docs = HTMLLoader().load(FIXTURES_DIR / "sample.html")
        content = docs[0].content
        # Script and style content should be gone
        assert "console.log" not in content
        assert "font-family" not in content
        # Nav and footer content should be gone
        assert "Copyright" not in content

    def test_html_headings_preserved(self):
        docs = HTMLLoader().load(FIXTURES_DIR / "sample.html")
        content = docs[0].content
        assert "Main Heading" in content
        assert "Sub Heading" in content

    def test_html_meaningful_content_kept(self):
        docs = HTMLLoader().load(FIXTURES_DIR / "sample.html")
        content = docs[0].content
        assert "first paragraph" in content
        assert "bold" in content

    def test_html_title_extraction(self):
        docs = HTMLLoader().load(FIXTURES_DIR / "sample.html")
        assert docs[0].metadata.title == "Sample HTML Page"

    def test_html_file_type(self):
        docs = HTMLLoader().load(FIXTURES_DIR / "sample.html")
        assert docs[0].metadata.file_type == "html"


# ---------------------------------------------------------------------------
# PDF loader tests
# ---------------------------------------------------------------------------

class TestPDFLoader:
    def test_load_pdf_pages(self):
        docs = PDFLoader().load(FIXTURES_DIR / "sample.pdf")
        assert len(docs) == 2  # 2-page PDF

    def test_pdf_page_numbers(self):
        docs = PDFLoader().load(FIXTURES_DIR / "sample.pdf")
        assert docs[0].metadata.page_number == 1
        assert docs[1].metadata.page_number == 2

    def test_pdf_total_pages(self):
        docs = PDFLoader().load(FIXTURES_DIR / "sample.pdf")
        for doc in docs:
            assert doc.metadata.total_pages == 2

    def test_pdf_content_per_page(self):
        docs = PDFLoader().load(FIXTURES_DIR / "sample.pdf")
        assert "Introduction" in docs[0].content or "Page 1" in docs[0].content
        assert "Hybrid" in docs[1].content or "Page 2" in docs[1].content

    def test_pdf_file_type(self):
        docs = PDFLoader().load(FIXTURES_DIR / "sample.pdf")
        for doc in docs:
            assert doc.metadata.file_type == "pdf"

    def test_pdf_deterministic_ids(self):
        docs1 = PDFLoader().load(FIXTURES_DIR / "sample.pdf")
        docs2 = PDFLoader().load(FIXTURES_DIR / "sample.pdf")
        ids1 = [d.metadata.document_id for d in docs1]
        ids2 = [d.metadata.document_id for d in docs2]
        assert ids1 == ids2


# ---------------------------------------------------------------------------
# DocumentLoader dispatcher tests
# ---------------------------------------------------------------------------

class TestDocumentLoader:
    def test_dispatch_txt(self):
        loader = DocumentLoader()
        docs = loader.load(FIXTURES_DIR / "sample.txt")
        assert len(docs) >= 1
        assert docs[0].metadata.file_type == "txt"

    def test_dispatch_md(self):
        loader = DocumentLoader()
        docs = loader.load(FIXTURES_DIR / "sample.md")
        assert len(docs) >= 1

    def test_dispatch_html(self):
        loader = DocumentLoader()
        docs = loader.load(FIXTURES_DIR / "sample.html")
        assert len(docs) >= 1

    def test_dispatch_pdf(self):
        loader = DocumentLoader()
        docs = loader.load(FIXTURES_DIR / "sample.pdf")
        assert len(docs) == 2

    def test_unsupported_extension(self, tmp_path):
        loader = DocumentLoader()
        p = tmp_path / "data.xyz"
        p.write_text("some content", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.load(p)

    def test_file_not_found(self):
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/doc.txt")

    def test_register_custom_loader(self, tmp_path):
        class CSVLoader(BaseLoader):
            def load(self, source):
                return [
                    RawDocument(
                        content="csv data",
                        metadata=DocumentMetadata(
                            source_file=str(source),
                            document_id="csv1",
                            file_type="csv",
                        ),
                    )
                ]

        loader = DocumentLoader()
        loader.register(".csv", CSVLoader())
        assert ".csv" in loader.supported_extensions

        p = tmp_path / "data.csv"
        p.write_text("a,b,c", encoding="utf-8")
        docs = loader.load(p)
        assert docs[0].metadata.file_type == "csv"

    def test_supported_extensions(self):
        loader = DocumentLoader()
        exts = loader.supported_extensions
        assert ".txt" in exts
        assert ".md" in exts
        assert ".html" in exts
        assert ".htm" in exts
        assert ".pdf" in exts


# ---------------------------------------------------------------------------
# Normalizer tests
# ---------------------------------------------------------------------------

class TestNormalizer:
    def test_normalize_text_whitespace(self):
        normalizer = DocumentNormalizer()
        text = "  Hello   world!  \n\n\n\nNew paragraph.  "
        result = normalizer.normalize_text(text)
        assert result == "Hello world!\n\nNew paragraph."

    def test_normalize_document(self):
        normalizer = DocumentNormalizer()
        raw = RawDocument(
            content="  Hello   world!  \n\n\n\nNew paragraph.  ",
            metadata=DocumentMetadata(
                source_file="test.txt", document_id="doc1", file_type="txt"
            ),
        )
        normalized = normalizer.normalize(raw)
        assert normalized.content == "Hello world!\n\nNew paragraph."
        assert normalized.metadata.document_id == "doc1"

    def test_normalize_empty_string(self):
        normalizer = DocumentNormalizer()
        assert normalizer.normalize_text("") == ""

    def test_normalize_control_characters(self):
        normalizer = DocumentNormalizer()
        text = "Hello\x00world\x07!"
        result = normalizer.normalize_text(text)
        assert "\x00" not in result
        assert "\x07" not in result
        assert "Helloworld!" in result


# ---------------------------------------------------------------------------
# Graceful error handling tests
# ---------------------------------------------------------------------------

class TestGracefulErrorHandling:
    def test_malformed_html(self, tmp_path):
        """Loader should not crash on broken HTML."""
        p = tmp_path / "broken.html"
        p.write_text("<html><body><p>Unclosed tag<p>More text", encoding="utf-8")
        docs = HTMLLoader().load(p)
        assert len(docs) == 1
        assert "Unclosed" in docs[0].content

    def test_empty_txt_file(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("", encoding="utf-8")
        docs = TextLoader().load(p)
        assert len(docs) == 1
        assert docs[0].content == ""

    def test_empty_md_file(self, tmp_path):
        p = tmp_path / "empty.md"
        p.write_text("", encoding="utf-8")
        docs = MarkdownLoader().load(p)
        assert len(docs) == 1


# ---------------------------------------------------------------------------
# CLI ingest_directory integration test
# ---------------------------------------------------------------------------

class TestIngestDirectory:
    def test_ingest_directory_e2e(self, tmp_path):
        """End-to-end test: load fixtures, normalize, write to output dir."""
        from scripts.ingest import ingest_directory

        output_dir = tmp_path / "processed"
        loader = DocumentLoader()
        normalizer = DocumentNormalizer()

        outputs = ingest_directory(FIXTURES_DIR, output_dir, loader, normalizer)
        assert len(outputs) > 0

        # Verify each output is valid JSON with expected keys
        for out_path in outputs:
            assert out_path.exists()
            data = json.loads(out_path.read_text(encoding="utf-8"))
            assert "content" in data
            assert "metadata" in data
            assert "document_id" in data["metadata"]
            assert "source_file" in data["metadata"]
            assert "ingestion_timestamp" in data["metadata"]

    def test_ingest_idempotent(self, tmp_path):
        """Re-running ingestion should produce identical output files."""
        from scripts.ingest import ingest_directory

        output1 = tmp_path / "run1"
        output2 = tmp_path / "run2"
        loader = DocumentLoader()
        normalizer = DocumentNormalizer()

        files1 = ingest_directory(FIXTURES_DIR, output1, loader, normalizer)
        files2 = ingest_directory(FIXTURES_DIR, output2, loader, normalizer)

        names1 = sorted(f.name for f in files1)
        names2 = sorted(f.name for f in files2)
        assert names1 == names2

    def test_ingest_empty_directory(self, tmp_path):
        from scripts.ingest import ingest_directory

        empty = tmp_path / "empty_input"
        empty.mkdir()
        output = tmp_path / "output"
        loader = DocumentLoader()
        normalizer = DocumentNormalizer()

        files = ingest_directory(empty, output, loader, normalizer)
        assert files == []
