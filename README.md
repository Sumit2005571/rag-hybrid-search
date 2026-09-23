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
   - Multi-format document loading (PDF, DOCX, TXT, Markdown).
   - Document metadata extraction and text normalization (whitespace, control characters).

2. **Chunking Strategies (`app/chunking`)**:
   - Fixed-size sliding window with configurable overlap.
   - Recursive structure-aware splitting using natural boundaries.
   - Semantic chunking using sentence distance metrics.

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
│   └── processed/             # Preprocessed & normalized documents
├── scripts/                   # Utility and server runner scripts
│   └── run_server.py
├── tests/                     # Unit and integration test suite
│   ├── __init__.py
│   ├── conftest.py            # Pytest fixtures and test environment
│   ├── test_api.py            # API endpoint tests
│   ├── test_config.py         # Configuration tests
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
