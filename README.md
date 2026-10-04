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
│   ├── retrieval/             # Dense, sparse, RRF, and cross-encoder reranking
│   │   ├── __init__.py
│   │   ├── dense.py
│   │   ├── sparse.py
│   │   ├── fusion.py
│   │   └── reranker.py
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
│   ├── test_api.py            # API endpoint tests
│   ├── test_config.py         # Configuration tests
│   ├── test_chunking.py       # Three-strategy chunking test suite (98 tests)
│   ├── test_ingestion.py      # Multi-format ingestion test suite
│   ├── test_logging.py        # Logging setup tests
│   └── test_structure.py     # Component & module tests
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

Start the FastAPI application via uvicorn or the runner script:

```bash
# Using Python runner script
python -m scripts.run_server

# Or directly with Uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## Running Tests

Run the test suite using `pytest`:

```bash
pytest
```

To run with verbose output:
```bash
pytest -v
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
