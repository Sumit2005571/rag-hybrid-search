"""CLI script to build or rebuild the BM25 sparse index.

Reads chunk JSON files produced by ``scripts/chunk.py``, then builds and
persists a BM25Okapi index to ``data/bm25/`` (configurable).

The BM25 index uses the SAME canonical chunk objects as the ChromaDB dense
index — the same chunk IDs, text, and metadata.  No re-chunking occurs.

Usage
-----
::

    # Build BM25 index from default chunks directory
    python scripts/index_bm25.py --input data/chunks

    # Custom index location
    python scripts/index_bm25.py --input data/chunks --index-dir data/bm25

    # With explicit top-k (for later search)
    python scripts/index_bm25.py --input data/chunks --top-k 20

All options
-----------
--input      Directory of chunk JSON files.   Default: data/chunks
--index-dir  BM25 persistence directory.      Default: from config (data/bm25)
--top-k      Default result count.            Default: from config (10)
--log-level  Logging verbosity.               Default: INFO

No OpenAI API key is required — BM25 is purely local.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chunking.base import Chunk
from app.config import get_settings
from app.logging_config import get_logger, setup_logging
from app.retrieval.sparse import BM25IndexError, BM25SparseStore

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_chunks_from_directory(input_dir: Path) -> List[Chunk]:
    """Load and validate all chunk JSON files from *input_dir*.

    Args:
        input_dir: Directory produced by ``scripts/chunk.py``.

    Returns:
        List of validated :class:`~app.chunking.base.Chunk` instances.
    """
    chunks: List[Chunk] = []
    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        logger.warning("No JSON files found in: %s", input_dir)
        return chunks

    for json_file in json_files:
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            chunk = Chunk.model_validate(data)
            chunks.append(chunk)
            logger.debug("Loaded chunk: %s", json_file.name)
        except Exception as exc:
            logger.error(
                "Failed to load chunk file '%s': %s — skipping.",
                json_file.name,
                exc,
            )

    return chunks


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Build/rebuild the BM25 sparse retrieval index from canonical chunks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/chunks",
        help="Directory containing chunk JSON files (output of scripts/chunk.py).",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Directory where the BM25 index will be persisted. "
             "Falls back to BM25_INDEX_DIRECTORY / config.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.top_k_sparse,
        help="Default number of results returned by BM25 search.",
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

    input_dir = Path(args.input)
    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Load chunks
    # -----------------------------------------------------------------------
    logger.info("Loading chunks from: %s", input_dir)
    chunks = load_chunks_from_directory(input_dir)

    if not chunks:
        logger.warning("No valid chunks found in '%s'. Nothing to index.", input_dir)
        sys.exit(0)

    logger.info("Loaded %d chunk(s).", len(chunks))

    # -----------------------------------------------------------------------
    # Build BM25 index
    # -----------------------------------------------------------------------
    index_dir = args.index_dir or settings.bm25_index_directory

    try:
        store = BM25SparseStore(index_directory=index_dir, top_k=args.top_k)
        indexed = store.build_index(chunks)
    except BM25IndexError as exc:
        logger.error("BM25 indexing failed: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Unexpected error during BM25 indexing: %s", exc)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("BM25 indexing summary")
    logger.info("  Chunks loaded   : %d", len(chunks))
    logger.info("  Chunks indexed  : %d", indexed)
    logger.info("  Index directory : %s", index_dir)
    logger.info("  Default top-k   : %d", args.top_k)
    logger.info("=" * 60)

    if indexed == 0:
        logger.error("No chunks were indexed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
