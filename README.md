# RAG Hybrid Search

A production-grade, modular Retrieval-Augmented Generation (RAG) system implementing hybrid search (dense embeddings + sparse BM25 lexical search), Reciprocal Rank Fusion (RRF), cross-encoder reranking, and citation-grounded LLM generation with confidence assessment and abstention.

---

## Architecture Overview

```
                      ┌───────────────────────┐
                      │    Incoming Query     │
                      └──────────┬────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       ┌───────────────────┐           ┌───────────────────┐
       │   Dense Search    │           │   Sparse Search   │
       │(ChromaDB + OpenAI)│           │      (BM25)       │
       └─────────┬─────────┘           └─────────┬─────────┘
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                     ┌──────────────────────┐
                     │ Reciprocal Rank      │
                     │ Fusion (RRF)         │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ Cross-Encoder        │
                     │ Reranker             │
                     └──────────┬───────────┘
                                │ Top-K Contexts
                                ▼
                     ┌──────────────────────┐
                     │ Grounded LLM         │
                     │ Generator (OpenAI)   │
                     └──────────┬───────────┘
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
       ┌───────────────────┐         ┌───────────────────┐
       │ Citation Analysis │         │ Confidence Scorer │
       │  & Verification   │         │   & Abstention    │
       └───────────────────┘         └───────────────────┘
```

The system is designed without high-level wrapper abstractions (such as LangChain) to maintain complete transparency, full control, and deterministic testability over each stage of the RAG pipeline.

### Core Components

1. **Ingestion & Normalization (`app/ingestion`)**:
   - Multi-format document loading (PDF, Markdown, HTML, TXT).
   - Document metadata extraction (`source_file`, `document_id`, `section_heading`, `page_number`, `ingestion_timestamp`).
   - Clean plaintext normalization (whitespace collapse, control character removal).
   - Deterministic SHA-256 document IDs enabling idempotent re-indexing.

2. **Chunking Strategies (`app/chunking`)**:
   - **Fixed-size** sliding window: configurable `chunk_size` and `overlap`, deterministic ordering, no missing or duplicated text.
   - **Recursive structure-aware**: respects Markdown heading → paragraph → sentence → word → character boundaries; prefer largest natural boundary first.
   - **Semantic**: detects topic boundaries via cosine-similarity of sentence embeddings; auto-falls back to paragraph splitting when the OpenAI key is absent or the API call fails.

3. **Embeddings & Vector Storage (`app/embeddings`, `app/retrieval/dense.py`)**:
   - OpenAI `text-embedding-3-small` vector generation.
   - ChromaDB local persistence and similarity search.

4. **Sparse Retrieval (`app/retrieval/sparse.py`)**:
   - In-memory BM25 lexical indexing using `rank-bm25`.

5. **Hybrid Search & Fusion (`app/retrieval/fusion.py`, `reranker.py`)**:
   - Reciprocal Rank Fusion ($RRF(d) = \sum \frac{1}{k + rank(d)}$).
   - Cross-encoder reranking with `sentence-transformers` for precise scoring.

6. **Generation & Verification (`app/generation`)**:
   - Grounded prompt templates enforcing document adherence.
   - Explicit inline citation tagging (`[Chunk: <id>]`) and verification against context.
   - Confidence scoring and automatic abstention ("I don't know") when threshold is not met.

7. **Evaluation (`app/evaluation`)**:
   - Automated evaluation suite covering faithfulness, answer relevance, context recall, and context precision.
   - Golden Q&A benchmark runner.

8. **API & Interface (`app/api`, `app/main.py`)**:
   - FastAPI REST API providing `/health`, `/api/v1/query`, and `/api/v1/ingest`.

---

## Directory Structure

```text
rag-hybrid-search/
├── app/
│   ├── __init__.py            # Package root & version
│   ├── config.py              # Environment configuration & Pydantic settings
│   ├── logging_config.py      # Standardized logging setup
│   ├── main.py                # FastAPI application entrypoint
│   ├── ingestion/             # Document loading, normalization, and metadata
│   │   ├── __init__.py
│   │   ├── loaders.py
│   │   ├── normalizer.py
│   │   └── metadata.py
│   ├── chunking/              # Fixed, recursive, and semantic chunking
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── fixed.py
│   │   ├── recursive.py
│   │   └── semantic.py
│   ├── embeddings/            # OpenAI dense embeddings
│   │   ├── __init__.py
│   │   └── openai_embeddings.py
│   ├── retrieval/             # Dense, sparse, hybrid RRF, and cross-encoder reranking
│   │   ├── __init__.py
│   │   ├── dense.py           # DenseRetriever — ChromaDB kNN search
│   │   ├── sparse.py          # BM25SparseStore — BM25 lexical index
│   │   ├── fusion.py          # ReciprocalRankFusion — RRF merging
│   │   ├── reranker.py        # CrossEncoderReranker — lazy-loaded CE model
│   │   ├── hybrid.py          # HybridRetriever — full pipeline orchestration
│   │   ├── indexer.py         # ChunkIndexer — embed + upsert to ChromaDB
│   │   └── vector_store.py    # VectorStore — ChromaDB abstraction
│   ├── generation/            # Prompts, generator, citations, verification, confidence
│   │   ├── __init__.py
│   │   ├── prompts.py
│   │   ├── generator.py
│   │   ├── citations.py
│   │   ├── verification.py
│   │   └── confidence.py
│   ├── evaluation/            # RAG evaluation metrics and benchmark runner
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── runner.py
│   └── api/                   # FastAPI routes and schemas
│       ├── __init__.py
│       ├── routes.py
│       └── schemas.py
├── data/                      # Local data directory (ignored by Git)
│   ├── raw/                   # Raw documents for ingestion
│   ├── processed/             # Preprocessed & normalized documents
│   └── chunks/                # Chunked documents (per-strategy output)
├── scripts/                   # Utility and runner scripts
│   ├── ingest.py              # CLI batch document ingestion script
│   ├── chunk.py               # CLI batch chunking script
│   └── run_server.py
├── tests/                     # Unit and integration test suite
│   ├── __init__.py
│   ├── conftest.py            # Pytest fixtures and test environment
│   ├── fixtures/              # Sample test documents (.txt, .md, .html, .pdf)
│   ├── test_bm25.py           # BM25 sparse retrieval test suite (24 tests)
│   ├── test_chunking.py       # Three-strategy chunking test suite (98 tests)
│   ├── test_hybrid.py         # Hybrid retrieval, RRF, reranker tests (44 tests)
│   ├── test_indexing.py       # Embeddings + ChromaDB indexing tests (44 tests)
│   ├── test_ingestion.py      # Multi-format ingestion test suite
│   ├── test_logging.py        # Logging setup tests
│   └── test_structure.py      # Component & module tests
├── .env.example               # Template environment configuration
├── .gitignore                 # Git ignore rules
├── pytest.ini                 # Pytest configuration
├── requirements.txt           # Project dependencies
└── README.md                  # System documentation
```

---

## Setup & Installation

### 1. Prerequisites
- Python 3.11+ (Python 3.13 supported)
- Git

### 2. Environment Setup

Clone the repository and set up a virtual environment:

```bash
# Clone repository
git clone https://github.com/Sumit2005571/rag-hybrid-search.git
cd rag-hybrid-search

# Create and activate virtual environment
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install pytest
```

### 4. Configure Environment Variables

Copy `.env.example` to `.env` and configure your API keys:

```bash
cp .env.example .env
```

Ensure your `OPENAI_API_KEY` is populated in `.env`:
```ini
OPENAI_API_KEY=sk-...
```

---

## Document Ingestion (Phase 1)

Phase 1 provides a multi-format, modular document ingestion pipeline that normalizes documents into clean plaintext with rich metadata preservation and deterministic ID generation for idempotent reprocessing.

### Supported Formats

| Format | Extension | Loader | Extracted Features & Metadata |
|---|---|---|---|
| **Plain Text** | `.txt` | `TextLoader` | Preserves paragraph boundaries, normalizes whitespace, UTF-8/Latin-1 fallback. |
| **Markdown** | `.md` | `MarkdownLoader` | Parses heading hierarchy (`#`, `##`, `###`), splits into section documents with `section_heading` preserved. |
| **HTML** | `.html`, `.htm` | `HTMLLoader` | Strips noise elements (`<script>`, `<style>`, `<nav>`, `<footer>`, `<header>`, `<aside>`, etc.), extracts `<title>`, preserves headings. |
| **PDF** | `.pdf` | `PDFLoader` | Extracts text page-by-page using PyMuPDF (`fitz`), tracking `page_number` (1-indexed) and `total_pages`. |

### Preserved Metadata Schema

Each ingested document is structured into a `RawDocument` containing:
- `source_file`: Absolute path of the source file.
- `document_id`: Deterministic 16-character SHA-256 hash based on `source_file` and page/section index.
- `file_type`: Format extension (`txt`, `md`, `html`, `pdf`).
- `title`: Document or page title where available.
- `section_heading`: Heading name for structured Markdown/HTML sections.
- `page_number`: 1-indexed page number for PDFs.
- `total_pages`: Total page count for paginated files.
- `ingestion_timestamp`: ISO-8601 UTC timestamp.
- `extra`: Arbitrary extensible metadata dictionary.

### CLI Batch Ingestion

Raw source documents are stored in `data/raw/` and processed JSON documents are saved to `data/processed/`.

```bash
# Ingest all supported documents from data/raw into data/processed
python scripts/ingest.py --input data/raw --output data/processed

# Verbose logging
python scripts/ingest.py --input data/raw --output data/processed --log-level DEBUG
```

Output files in `data/processed/` are named `<document_id>.json` formatted as:
```json
{
  "content": "Normalized text content...",
  "metadata": {
    "source_file": "...",
    "document_id": "...",
    "file_type": "...",
    "page_number": 1,
    "section_heading": null,
    "ingestion_timestamp": "..."
  }
}
```

Because IDs are deterministic, re-running ingestion reprocesses the raw documents in-place without duplicate artifacts or requiring re-uploading.

### Programmatic Usage

```python
from pathlib import Path
from app.ingestion.loaders import DocumentLoader
from app.ingestion.normalizer import DocumentNormalizer

loader = DocumentLoader()
normalizer = DocumentNormalizer()

# Load and normalize any supported file
raw_docs = loader.load(Path("data/raw/sample.pdf"))
for raw_doc in raw_docs:
    clean_doc = normalizer.normalize(raw_doc)
    print(f"[{clean_doc.metadata.file_type}] Page {clean_doc.metadata.page_number}: {clean_doc.content[:100]}...")
```

---

## Chunking Strategies (Phase 1B)

The chunking pipeline converts ingested `data/processed/*.json` documents into
fixed-size text windows suitable for embedding and retrieval.  Three strategies
are provided; all implement the same `BaseChunker` interface and produce
consistent `Chunk` objects.

### Strategy A — Fixed-Size (`fixed`)

Slides a window of exactly `chunk_size` characters across the document,
advancing by `chunk_size − overlap` characters per step.  Every character
appears in at least one chunk; the overlap region appears in exactly two
consecutive chunks.

```
|<------ chunk_size ------>|
        |<------ chunk_size ------>|
|<-ov-->|                 |<-ov-->|
```

**Best for:** Documents without clear structure where uniform context windows
are desired.

### Strategy B — Recursive Structure-Aware (`recursive`)

Tries separators in priority order:
1. `\n## ` / `\n# ` (Markdown headings)
2. `\n\n` (paragraph breaks)
3. `\n` (line breaks)
4. `". "` / `"? "` / `"! "` (sentence boundaries)
5. `" "` (word boundary)
6. `""` (hard character split — last resort)

Any segment that still exceeds `chunk_size` after the current separator is
recursively split with the next separator in the list.

**Best for:** Structured documents (Markdown, reports) where respecting section
boundaries preserves semantic coherence.

### Strategy C — Semantic (`semantic`)

1. Splits the document into sentences.
2. Embeds each sentence via the configured OpenAI model (`text-embedding-3-small`).
3. Computes cosine similarity between consecutive sentence embeddings.
4. Declares a chunk boundary wherever similarity drops below `SEMANTIC_BREAKPOINT_THRESHOLD`.
5. Enforces `SEMANTIC_MIN_CHUNK_SIZE` (merge short chunks) and `SEMANTIC_MAX_CHUNK_SIZE` (re-split long chunks).
6. **Graceful fallback:** If no API key is set or the embedding call fails, the
   strategy transparently falls back to paragraph-boundary splitting.

**Best for:** Long-form text where topic shifts don't align with structural markers.

### Configuration

All settings can be overridden via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_CHUNK_SIZE` | `500` | Target max characters per chunk (fixed & recursive) |
| `DEFAULT_CHUNK_OVERLAP` | `50` | Character overlap between consecutive chunks |
| `SEMANTIC_BREAKPOINT_THRESHOLD` | `0.75` | Cosine-similarity threshold for topic boundary detection |
| `SEMANTIC_MIN_CHUNK_SIZE` | `100` | Minimum characters per semantic chunk (smaller chunks are merged) |
| `SEMANTIC_MAX_CHUNK_SIZE` | `2000` | Hard maximum characters per semantic chunk |

### CLI Batch Chunking

First ingest documents, then chunk them:

```bash
# 1. Ingest raw documents
python scripts/ingest.py --input data/raw --output data/processed

# 2. Chunk with fixed-size strategy (default params from Settings)
python scripts/chunk.py --strategy fixed

# 3. Chunk with recursive strategy, custom parameters
python scripts/chunk.py --strategy recursive --chunk-size 300 --overlap 30

# 4. Chunk with semantic strategy
python scripts/chunk.py --strategy semantic --threshold 0.7

# 5. Custom input/output directories
python scripts/chunk.py --strategy fixed --input data/processed --output data/chunks

# 6. Verbose logging
python scripts/chunk.py --strategy recursive --log-level DEBUG
```

### Chunk Output Format

Each chunk is written as `data/chunks/<chunk_id>.json`:

```json
{
  "chunk_id": "abc123#c0",
  "document_id": "abc123",
  "source_file": "/abs/path/to/document.pdf",
  "chunk_index": 0,
  "start_char": 0,
  "end_char": 487,
  "text": "This is the first chunk of text...",
  "char_count": 487,
  "section_heading": "## Introduction",
  "page_number": 1,
  "strategy": "fixed",
  "metadata": {}
}
```

### Programmatic Usage

```python
from app.chunking import FixedSizeChunker, RecursiveStructureChunker, SemanticChunker

# Fixed-size
chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=50)
chunks = chunker.chunk(
    text=doc.content,
    document_id=doc.metadata.document_id,
    metadata={"source_file": doc.metadata.source_file, "page_number": doc.metadata.page_number},
)

# Recursive
chunker = RecursiveStructureChunker(chunk_size=500, chunk_overlap=50)

# Semantic (with injected embedding function for tests)
chunker = SemanticChunker(
    breakpoint_threshold=0.75,
    embedding_fn=my_embed_fn,   # optional; uses OpenAI by default
)
```

---

## Running the API Server

> **Prerequisites before starting the server:**
> 1. `OPENAI_API_KEY` must be set in your `.env` file.
> 2. The ChromaDB index must be populated (run the ingestion and chunking scripts first).
> 3. The BM25 index must exist at `BM25_INDEX_DIRECTORY` (built during indexing).
>
> The server will start without these, but `/api/v1/query` will return HTTP 503
> until both indexes are available.

Start the FastAPI application via uvicorn:

```bash
# Development (auto-reload)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

Interactive API documentation will be available at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## Phase 3 — End-to-End Query Flow (This Milestone)

Phase 3 wires the full RAG pipeline into the `/api/v1/query` endpoint:
hybrid retrieval → grounded LLM generation → citation extraction & verification → confidence assessment.

### Request-to-Answer Flow

```
POST /api/v1/query
  │
  ├─ 1. Validate request (Pydantic)
  ├─ 2. Build HybridRetriever
  │       ├─ EmbeddingGenerator (OpenAI text-embedding-3-small)
  │       ├─ DenseRetriever (ChromaDB kNN)
  │       └─ BM25SparseStore (persisted BM25 index)
  ├─ 3. retrieve(query, top_k, use_reranker)
  │       ├─ Dense search → ChromaDB
  │       ├─ Sparse search → BM25
  │       ├─ RRF fusion
  │       └─ Cross-encoder reranking (optional)
  ├─ 4. RAGGenerator.generate(question, retrieved_contexts)
  │       ├─ Build grounded prompt (system + user)
  │       ├─ openai.ChatCompletion.create (gpt-4o-mini)
  │       ├─ CitationExtractor — find [Chunk: <id>] markers
  │       ├─ CitationVerifier — drop markers not in retrieved set
  │       └─ ConfidenceScorer — avg retrieval score vs threshold
  └─ 5. Return QueryResponse
          ├─ answer (LLM text)
          ├─ citations (CitationItem[]: chunk_id, source, text_snippet)
          ├─ confidence_score (avg retrieval relevance)
          ├─ abstained (bool)
          ├─ verification_score (fraction of valid citations)
          └─ retrieved_chunks (full chunk list for inspection)
```

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | **Yes** | — | OpenAI key for embeddings and chat completion |
| `OPENAI_EMBEDDING_MODEL` | No | `text-embedding-3-small` | Embedding model |
| `OPENAI_CHAT_MODEL` | No | `gpt-4o-mini` | Chat completion model |
| `CHROMA_PERSIST_DIRECTORY` | No | `./data/chroma` | ChromaDB data directory |
| `CHROMA_COLLECTION_NAME` | No | `rag_documents` | ChromaDB collection |
| `BM25_INDEX_DIRECTORY` | No | `./data/bm25` | BM25 index directory |
| `TOP_K_DENSE` | No | `10` | Dense retrieval candidates |
| `TOP_K_SPARSE` | No | `10` | Sparse retrieval candidates |
| `TOP_K_FINAL` | No | `5` | Final results after reranking |
| `CONFIDENCE_THRESHOLD` | No | `0.7` | Abstention threshold (avg retrieval score) |
| `CROSS_ENCODER_MODEL` | No | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking model |

### Indexing Prerequisites

Before calling `/api/v1/query` with real data:

```bash
# Step 1 — Ingest raw documents
python scripts/ingest.py --input data/raw --output data/processed

# Step 2 — Chunk documents
python scripts/chunk.py --strategy fixed

# Step 3 — Embed and index into ChromaDB + build BM25 index
#   (Uses the ChunkIndexer; run after chunking)
python -c "
from app.retrieval.indexer import ChunkIndexer
indexer = ChunkIndexer()
indexer.index_from_directory('data/chunks')
"
```

> **Note:** Dense retrieval requires a populated ChromaDB collection.  Sparse
> retrieval requires a persisted BM25 index.  If only one index is available,
> the pipeline degrades gracefully: failed retrieval legs return empty lists and
> the fusion step works with whatever is available.

### Example API Requests

**Health check:**
```bash
curl http://localhost:8000/health
```

**Query (grounded answer):**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Reciprocal Rank Fusion?",
    "top_k": 5,
    "use_reranker": true
  }'
```

**Example response (with populated index):**
```json
{
  "query": "What is Reciprocal Rank Fusion?",
  "answer": "Reciprocal Rank Fusion (RRF) is a score fusion method ... [Chunk: doc1#c3]",
  "citations": [
    {
      "chunk_id": "doc1#c3",
      "source": "rag_overview.pdf",
      "text_snippet": "RRF combines ranked lists by summing 1/(k + rank) ..."
    }
  ],
  "confidence_score": 0.87,
  "abstained": false,
  "verification_score": 1.0,
  "retrieved_chunks": [
    {
      "chunk_id": "doc1#c3",
      "document_id": "doc1",
      "text": "RRF combines ranked lists ...",
      "score": 0.87,
      "rank": 1,
      "metadata": {"source_file": "rag_overview.pdf"}
    }
  ]
}
```

**Abstention (when context is insufficient):**
```json
{
  "query": "What is the population of Mars?",
  "answer": "I don't know based on the provided documents.",
  "citations": [],
  "confidence_score": 0.0,
  "abstained": true,
  "verification_score": null,
  "retrieved_chunks": []
}
```

### Abstention and Confidence

The `confidence_score` is the **average retrieval relevance score** of the
returned chunks, not a calibrated probability of correctness.  The system
abstains (`abstained: true`) when:

1. No chunks are retrieved (score = 0).
2. The average retrieval score is below `CONFIDENCE_THRESHOLD` (default 0.7).
3. The generated answer contains uncertainty phrases like "I don't know".
4. The OpenAI API key is missing or the API call fails.

### Citation Verification

Citations are extracted from the model's output using the `[Chunk: <id>]` marker
format.  Each cited chunk ID is checked against the set of actually retrieved
chunks.  Citations to non-retrieved chunks (hallucinations) are **removed
before the response is returned**.  The `verification_score` field reports the
fraction of extracted citations that passed verification (1.0 = all citations
grounded, 0.0 = all hallucinated).

> **Unverified integration note:** Live end-to-end generation requires a valid
> `OPENAI_API_KEY` and a populated index.  All tests in this milestone use mocks
> and do not call the real OpenAI API or real ChromaDB.

---

## Running Tests

Run the full test suite using `pytest` (304 tests, no API key or network required):

```bash
# Windows (venv)
.\venv\Scripts\python.exe -m pytest tests/ -v

# Linux / macOS
python -m pytest tests/ -v
```

All tests use mocks — no OpenAI credits, no real cross-encoder download, no live ChromaDB writes outside `tmp_path`.

Key test files:

| File | Description | Tests |
|---|---|---|
| `test_query_flow.py` | Full /query pipeline, RAGGenerator, citations, abstention | 34 |
| `test_hybrid.py` | Hybrid retrieval, RRF, cross-encoder reranker | 44 |
| `test_indexing.py` | Embeddings + ChromaDB indexing | 44 |
| `test_bm25.py` | BM25 sparse retrieval | 24 |
| `test_chunking.py` | Three-strategy chunking | 98 |
| `test_api.py` | FastAPI endpoint smoke tests | 4 |

To run only the query-flow test suite:

```bash
.\venv\Scripts\python.exe -m pytest tests/test_query_flow.py -v
```

To run only the hybrid-retrieval test suite:

```bash
.\venv\Scripts\python.exe -m pytest tests/test_hybrid.py -v
```

---

## Security & Best Practices

- **Zero Secrets in Git**: Secrets, `.env` files, local SQLite/Chroma databases, and raw document directories are excluded via `.gitignore`.
- **Modular Design**: Every component implements typed interfaces and can be unit tested in isolation without external API dependencies.
- **Fail-Safe Generation**: Built-in verification and abstention mechanisms prevent ungrounded hallucinations from being served to end-users.

---

## Phase 2.1 — Embeddings and ChromaDB

### Why embeddings?

Dense vector embeddings convert text into high-dimensional numerical vectors where semantically similar passages map to nearby points in vector space.  This enables semantic search: a query about "machine learning" can retrieve chunks about "neural networks" even when none of those exact words appear in the query — something keyword search cannot do.

### Why ChromaDB?

[ChromaDB](https://www.trychroma.com/) is an open-source, embeddable vector database that:
- Stores embeddings and metadata together with full persistence.
- Supports fast approximate nearest-neighbour search.
- Runs entirely locally — no external service required.
- Survives application restarts via its persistent storage backend.

### Embedding model

| Setting | Value |
|---|---|
| Model | `text-embedding-3-small` |
| Provider | OpenAI |
| Dimensions | 1536 |
| Batching | Configurable (default 100 chunks per API call) |

### Configuration

All settings can be provided via environment variables or a `.env` file:

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key for embedding calls |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name |
| `CHROMA_PERSIST_DIRECTORY` | `./data/chroma` | Where ChromaDB persists its data |
| `CHROMA_COLLECTION_NAME` | `rag_documents` | ChromaDB collection to use |
| `EMBEDDING_BATCH_SIZE` | `100` | Chunks sent per embedding API request |

### Setting OPENAI_API_KEY

1. Copy the example env file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and replace the placeholder:
   ```
   OPENAI_API_KEY=sk-your-real-key-here
   ```

3. The application reads this file automatically on startup via `python-dotenv`.

> **Never commit real API keys.** `.env` is listed in `.gitignore`.

### How persistent storage works

ChromaDB writes its data files under `data/chroma/` (configurable).  Once the indexing script has run, this directory contains a fully queryable vector store that survives application restarts.  The `data/chroma/` path is excluded from Git to prevent committing large binary files.

### How to run the indexing CLI

After ingesting and chunking your documents:

```bash
# Step 1 – Ingest raw documents
python scripts/ingest.py --input data/raw --output data/processed

# Step 2 – Chunk them (choose a strategy)
python scripts/chunk.py --strategy fixed --input data/processed --output data/chunks

# Step 3 – Embed and index into ChromaDB
python scripts/index.py --input data/chunks
```

**Example with all options:**

```bash
python scripts/index.py \
    --input data/chunks \
    --chroma-dir data/chroma \
    --collection rag_documents \
    --model text-embedding-3-small \
    --batch-size 50 \
    --log-level INFO
```

The script will:
1. Load all chunk JSON files from `--input`.
2. Validate them against the `Chunk` schema.
3. Batch-embed chunk text via the OpenAI API.
4. Upsert all chunks into the ChromaDB collection.
5. Print a progress + summary report.
6. Exit with code `0` on full success, `1` on partial or total failure.

### How idempotent indexing works

Chunks are identified by a **deterministic ID** (e.g. `abc123#c5`) derived from the document ID, chunk strategy, and chunk position.  ChromaDB's `upsert` operation is used for all writes.

Running the indexing script on the same corpus twice:
- Generates the **same IDs** for the same chunks.
- **Overwrites** existing records rather than creating duplicates.
- Results in the same collection count after both runs.

### Running the tests

```bash
# All tests (unit + integration, no live API required)
python -m pytest

# Phase 2.1 tests only
python -m pytest tests/test_indexing.py -v

# Verbose output for the full suite
python -m pytest -v
```

All Phase 2.1 tests mock the OpenAI embedding API — no real API key is needed to run them.  Live integration tests (if added later) should be placed in a separate `tests/integration/` directory and clearly labelled.

---

## Phase 2.2 — BM25 Sparse Retrieval

### What is BM25?

BM25 (Best Matching 25) is a sparse retrieval algorithm that scores documents based on exact keyword matches. Unlike dense embeddings (which understand meaning/semantics), BM25 relies on exact lexical matches.

### Why BM25 complements dense retrieval

Dense retrieval can struggle with:
- Exact part numbers or error codes (e.g., `HTTP 500`)
- Variable names or configuration keys (e.g., `DATABASE_URL`)
- Function signatures (e.g., `get_user_by_id`)

BM25 solves this by excelling at exact technical keyword searches. In Phase 2.3, the results of both retrievers will be fused together.

### Implementation Details

- **Canonical Chunks**: BM25 uses the *exact same* chunk JSON objects as the ChromaDB indexing pipeline. No re-chunking occurs.
- **Persistence**: The index is stored locally as a JSON metadata file and a Pickled model in `./data/bm25/`.
- **Idempotency**: Running the index builder multiple times performs a deterministic full rebuild, replacing the existing index and avoiding duplicate records.
- **Tokenization**: Our custom tokenizer splits on punctuation but deliberately preserves underscores and dots for technical identifiers (`HTTP_500`, `v1.2.3`).

> **Updated in Phase 2.3**: Dense retrieval, BM25 sparse retrieval, Reciprocal Rank Fusion (RRF), and cross-encoder reranking are now fully integrated via ``HybridRetriever``.

### How to use BM25

1. Generate your processed chunks first (see Phase 2.1).
2. Build the BM25 index:

```bash
python scripts/index_bm25.py --input data/chunks
```

3. Search the index:

```bash
python scripts/search_bm25.py --query "DATABASE_URL" --top-k 5
```


---

## Phase 2.3 — Hybrid Retrieval, RRF, and Cross-Encoder Reranking

### Overview

Phase 2.3 completes the retrieval stage by wiring dense and sparse search
together into a unified hybrid pipeline with two-stage ranking:

1. **Dense retrieval** — ChromaDB kNN over OpenAI embeddings (`DenseRetriever`)
2. **Sparse retrieval** — BM25 lexical keyword search (`BM25SparseStore`)
3. **Reciprocal Rank Fusion** — merges the two ranked lists, deduplicates shared
   chunks, and produces a single RRF-scored list (`ReciprocalRankFusion`)
4. **Cross-encoder reranking** — each (query, chunk-text) pair is scored by a
   local `sentence-transformers` cross-encoder model; results are re-ordered by
   cross-encoder relevance (`CrossEncoderReranker`)

### Key design decisions

| Decision | Rationale |
|---|---|
| **Lazy model loading** | The cross-encoder is only loaded on the first `rerank()` call, so importing the package does not trigger a model download. |
| **No silent fallback** | If the cross-encoder fails to load or score, `CrossEncoderError` is raised; callers choose whether to fall back to RRF ordering. |
| **Unified result type** | `RetrievalResult` is shared by dense, sparse, RRF, and reranker paths — no conversion needed. |
| **Graceful degradation** | `HybridRetriever` catches dense/sparse failures individually; if one path fails, the other still contributes. |

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `TOP_K_DENSE` | `10` | Candidates from dense retrieval |
| `TOP_K_SPARSE` | `10` | Candidates from sparse BM25 |
| `TOP_K_FINAL` | `5` | Final results after reranking |
| `RRF_K` | `60` | RRF constant *k* |
| `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Hugging Face cross-encoder |

### End-to-end pipeline (full workflow)

```bash
# 1. Ingest raw documents
python scripts/ingest.py --input data/raw --output data/processed

# 2. Chunk documents
python scripts/chunk.py --strategy fixed --input data/processed --output data/chunks

# 3. Embed + index into ChromaDB (requires OPENAI_API_KEY)
python scripts/index.py --input data/chunks

# 4. Build the BM25 sparse index
python scripts/index_bm25.py --input data/chunks

# 5. Run the test suite (no network or API key needed)
.\\venv\\Scripts\\python.exe -m pytest tests/ -v
```

### Programmatic usage

```python
from app.retrieval import (
    DenseRetriever,
    HybridRetriever,
    CrossEncoderReranker,
)
from app.retrieval.sparse import BM25SparseStore
from app.embeddings.embeddings import EmbeddingGenerator

# Set up components
embedding_gen = EmbeddingGenerator(api_key="sk-...")
dense = DenseRetriever()          # uses config defaults for ChromaDB path
sparse = BM25SparseStore()
sparse.load_index()               # loads from data/bm25/

reranker = CrossEncoderReranker() # loads lazily on first rerank() call

retriever = HybridRetriever(
    dense_retriever=dense,
    sparse_store=sparse,
    embedding_generator=embedding_gen,
    reranker=reranker,
)

results = retriever.retrieve("What is DATABASE_URL?", top_k=5)
for r in results:
    print(f"[{r.score:.4f}] {r.chunk_id}: {r.text[:80]}")
```

### Test suite (44 tests — all mocked)

| Class | Coverage |
|---|---|
| `TestReciprocalRankFusion` | RRF ordering, score formula, duplicates, empty lists, metadata preservation |
| `TestCrossEncoderReranker` | Lazy load, score ordering, top_k, load failure, inference failure, cached errors |
| `TestDenseRetriever` | Empty collection, result fields, score conversion, top_k guards |
| `TestHybridRetriever` | Integration, empty query, use_reranker flag, graceful degradation, metadata |

```bash
.\\venv\\Scripts\\python.exe -m pytest tests/test_hybrid.py -v
```
