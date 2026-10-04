"""CLI script to search the BM25 sparse index.

Loads the persisted BM25 index from ``data/bm25/`` (configurable) and
runs a BM25 sparse search for the given query.  No OpenAI API key is required.

Usage
-----
::

    # Basic search
    python scripts/search_bm25.py --query "DATABASE_URL"

    # Return more results
    python scripts/search_bm25.py --query "get_user_by_id" --top-k 20

    # Custom index location
    python scripts/search_bm25.py --query "HTTP 500" --index-dir data/bm25

    # Quiet output (just results)
    python scripts/search_bm25.py --query "MAX_RETRIES" --log-level WARNING

All options
-----------
--query      Search query string.                     Required.
--top-k      Number of results to return.             Default: from config (10)
--index-dir  BM25 persistence directory.              Default: from config
--log-level  Logging verbosity.                       Default: INFO

Exit codes
----------
0   Search completed successfully (even if 0 results found).
1   Index not found, query error, or unrecoverable failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.logging_config import get_logger, setup_logging
from app.retrieval.sparse import BM25IndexError, BM25SparseStore

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Search the BM25 sparse index for a query.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query string.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.top_k_sparse,
        help="Number of results to return.",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="BM25 index directory. Falls back to BM25_INDEX_DIRECTORY / config.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    index_dir = args.index_dir or settings.bm25_index_directory

    # -----------------------------------------------------------------------
    # Load index
    # -----------------------------------------------------------------------
    store = BM25SparseStore(index_directory=index_dir, top_k=args.top_k)
    try:
        count = store.load_index()
        logger.info("BM25 index loaded: %d chunk(s) from '%s'", count, index_dir)
    except BM25IndexError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------------
    query = args.query.strip()
    logger.info("Searching for: '%s' (top_k=%d)", query, args.top_k)

    try:
        results = store.search(query=query, top_k=args.top_k)
    except ValueError as exc:
        logger.error("Invalid query: %s", exc)
        sys.exit(1)
    except BM25IndexError as exc:
        logger.error("Search failed: %s", exc)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Display results
    # -----------------------------------------------------------------------
    print(f"\nBM25 Search Results for: '{query}'")
    print(f"{'-' * 60}")

    if not results:
        print("  No results found.")
    else:
        for result in results:
            print(f"\n  Rank #{result.rank}  |  Score: {result.score:.4f}")
            print(f"  Chunk ID   : {result.chunk_id}")
            print(f"  Document ID: {result.document_id}")
            src = result.metadata.get("source_file", "")
            if src:
                print(f"  Source     : {src}")
            strategy = result.metadata.get("chunking_strategy", "")
            if strategy:
                print(f"  Strategy   : {strategy}")
            page = result.metadata.get("page_number", "")
            if page:
                print(f"  Page       : {page}")
            heading = result.metadata.get("section_heading", "")
            if heading:
                print(f"  Heading    : {heading}")
            # Show first 200 chars of text
            preview = result.text[:200].replace("\n", " ")
            if len(result.text) > 200:
                preview += "..."
            print(f"  Text       : {preview}")

    print(f"\n{'-' * 60}")
    print(f"  Total results: {len(results)}")
    print()


if __name__ == "__main__":
    main()
