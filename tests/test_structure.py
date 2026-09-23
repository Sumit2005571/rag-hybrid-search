"""Unit tests verifying architectural components and modules are properly importable."""

from app.ingestion import DocumentMetadata, DocumentNormalizer, DocumentLoader
from app.ingestion.metadata import RawDocument
from app.chunking import BaseChunker, Chunk, FixedSizeChunker, RecursiveStructureChunker, SemanticChunker
from app.embeddings import OpenAIEmbeddingGenerator
from app.retrieval import DenseRetriever, SparseRetriever, ReciprocalRankFusion, CrossEncoderReranker, RetrievalResult
from app.generation import (
    build_rag_prompt,
    Citation,
    CitationExtractor,
    CitationVerifier,
    ConfidenceScorer,
    RAGGenerator,
)
from app.evaluation import EvaluationMetrics, RAGEvaluator, EvaluationRunner


def test_chunking_fixed():
    """Verify fixed-size chunking divides text and sets metadata."""
    chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=2)
    text = "Hello world, this is a test."
    chunks = chunker.chunk(text=text, document_id="doc1")
    assert len(chunks) > 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].chunk_id == "doc1#c0"


def test_normalizer():
    """Verify document normalization strips extra spaces."""
    normalizer = DocumentNormalizer()
    raw = RawDocument(
        content="  Hello   world!  \n\n\n\nNew paragraph.  ",
        metadata=DocumentMetadata(source="test.txt", document_id="doc1"),
    )
    normalized = normalizer.normalize(raw)
    assert normalized.content == "Hello world!\n\nNew paragraph."


def test_rrf_fusion():
    """Verify reciprocal rank fusion ranks correctly."""
    rrf = ReciprocalRankFusion(k=60)
    list1 = [
        RetrievalResult(chunk_id="c1", text="text 1", score=0.9),
        RetrievalResult(chunk_id="c2", text="text 2", score=0.8),
    ]
    list2 = [
        RetrievalResult(chunk_id="c2", text="text 2", score=0.95),
        RetrievalResult(chunk_id="c3", text="text 3", score=0.7),
    ]
    fused = rrf.fuse([list1, list2], top_k=3)
    assert len(fused) == 3
    # c2 appears in both lists, should have highest score
    assert fused[0].chunk_id == "c2"


def test_citation_extractor_and_verifier():
    """Verify citation extraction and validation."""
    extractor = CitationExtractor()
    sample_text = "Claim supported here [Chunk: doc1#c0] and another [Chunk: doc1#c1]."
    citations = extractor.extract(sample_text)
    assert citations == ["doc1#c0", "doc1#c1"]

    verifier = CitationVerifier()
    available = {"doc1#c0": "context 0"}
    res = verifier.verify(citations, available)
    assert not res.is_verified
    assert res.valid_citations == ["doc1#c0"]
    assert res.hallucinated_citations == ["doc1#c1"]
    assert res.verification_score == 0.5
