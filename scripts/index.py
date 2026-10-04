"""CLI script for embedding and indexing chunked documents into ChromaDB.

Reads chunk JSON files produced by ``scripts/chunk.py`` (or any directory of
per-chunk JSON files matching the :class:`~app.chunking.base.Chunk` schema),
generates embeddings via OpenAI, and upserts them into a persistent ChromaDB
collection.

Usage
-----
::

    # Index all chunks in data/chunks/ (default output of scripts/chunk.py)
    python scripts/index.py --input data/chunks

    # Custom ChromaDB location and collection
    python scripts/index.py --input data/chunks \\
        --chroma-dir data/my_chroma \\
        --collection my_collection

    # Control embedding model and batch size
    python scripts/index.py --input data/chunks \\
        --model text-embedding-3-small \\
        --batch-size 50

    # Run from the project root after chunking
    python scripts/chunk.py --strategy fixed
    python scripts/index.py --input data/chunks

All options
-----------
--input         Directory of chunk JSON files.       Default: data/chunks
--chroma-dir    ChromaDB persistence directory.      Default: from config
--collection    ChromaDB collection name.            Default: from config
--model         OpenAI embedding model.              Default: from config
--batch-size    Chunks per embedding API request.    Default: 100
--log-level     Logging verbosity.                   Default: INFO

Environment variables
---------------------
OPENAI_API_KEY           Required.
OPENAI_EMBEDDING_MODEL   Override default embedding model.
CHROMA_PERSIST_DIRECTORY Override default ChromaDB directory.
CHROMA_COLLECTION_NAME   Override default collection name.
EMBEDDING_BATCH_SIZE     Override default batch size.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

# Ensure project root is importable when running as a top-level script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chunking.base import Chunk
from app.config import get_settings
from app.embeddings.embeddings import EmbeddingConfigError, EmbeddingGenerator
from app.logging_config import get_logger, setup_logging
from app.retrieval.indexer import ChunkIndexer
from app.retrieval.vector_store import VectorStore, VectorStoreError

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_chunks_from_directory(input_dir: Path) -> List[Chunk]:
    """Load all chunk JSON files from *input_dir* and return :class:`Chunk` objects.

    Args:
        input_dir: Directory produced by ``scripts/chunk.py``.

    Returns:
        Ordered list of :class:`Chunk` instances.
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
            logger.error("Failed to load chunk file '%s': %s — skipping.", json_file.name, exc)

    return chunks


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Embed chunks and index them into persistent ChromaDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/chunks",
        help="Directory containing chunk JSON files (output of scripts/chunk.py).",
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default=None,
        help="ChromaDB persistence directory. Falls back to CHROMA_PERSIST_DIRECTORY / config.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="ChromaDB collection name. Falls back to CHROMA_COLLECTION_NAME / config.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenAI embedding model. Falls back to OPENAI_EMBEDDING_MODEL / config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("EMBEDDING_BATCH_SIZE", "100")),
        help="Number of chunks to embed in a single API request.",
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
    # Initialise embedding generator
    # -----------------------------------------------------------------------
    model = args.model or os.getenv("OPENAI_EMBEDDING_MODEL") or settings.openai_embedding_model
    try:
        embedding_generator = EmbeddingGenerator(
            model=model,
            batch_size=args.batch_size,
            validate_on_init=True,
        )
    except EmbeddingConfigError as exc:
        logger.error("Embedding configuration error: %s", exc)
        sys.exit(1)

    logger.info(
        "Embedding generator ready: model=%s, batch_size=%d",
        embedding_generator.model,
        embedding_generator.batch_size,
    )

    # -----------------------------------------------------------------------
    # Initialise vector store
    # -----------------------------------------------------------------------
    chroma_dir = args.chroma_dir or settings.chroma_persist_directory
    collection_name = args.collection or settings.chroma_collection_name

    try:
        vector_store = VectorStore(
            persist_directory=chroma_dir,
            collection_name=collection_name,
        )
    except VectorStoreError as exc:
        logger.error("ChromaDB initialisation failed: %s", exc)
        sys.exit(1)

    logger.info(
        "Vector store ready: collection='%s', persist_dir='%s', existing_chunks=%d",
        collection_name,
        chroma_dir,
        vector_store.count(),
    )

    # -----------------------------------------------------------------------
    # Index chunks
    # -----------------------------------------------------------------------
    indexer = ChunkIndexer(
        embedding_generator=embedding_generator,
        vector_store=vector_store,
    )

    try:
        indexed = indexer.index_chunks(chunks)
    except Exception as exc:
        logger.error("Indexing failed with an unexpected error: %s", exc)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    total_in_collection = vector_store.count()
    logger.info("=" * 60)
    logger.info("Indexing summary")
    logger.info("  Chunks loaded  : %d", len(chunks))
    logger.info("  Chunks indexed : %d", indexed)
    logger.info("  Failed / skipped: %d", len(chunks) - indexed)
    logger.info("  Total in collection: %d", total_in_collection)
    logger.info("  ChromaDB path  : %s", chroma_dir)
    logger.info("  Collection     : %s", collection_name)
    logger.info("=" * 60)

    if indexed < len(chunks):
        logger.warning(
            "%d chunk(s) could not be indexed. Check the logs above for details.",
            len(chunks) - indexed,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
