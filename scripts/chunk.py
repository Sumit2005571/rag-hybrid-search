"""CLI script for chunking processed documents.

Reads normalized JSON documents from ``data/processed/`` (produced by
``scripts/ingest.py``), applies the selected chunking strategy, and writes
the resulting chunks as JSON files into an output directory.

Usage
-----
::

    # Fixed-size chunking (default)
    python scripts/chunk.py --strategy fixed

    # Recursive structure-aware chunking
    python scripts/chunk.py --strategy recursive

    # Semantic chunking (requires OPENAI_API_KEY; falls back to paragraphs otherwise)
    python scripts/chunk.py --strategy semantic

    # Custom parameters
    python scripts/chunk.py --strategy fixed --chunk-size 300 --overlap 30

    # Custom directories
    python scripts/chunk.py --strategy recursive --input data/processed --output data/chunks

All options
-----------
--strategy      {fixed,recursive,semantic}   Chunking strategy (required).
--input         Directory of processed JSON documents.  Default: data/processed
--output        Directory to write chunk JSON files.    Default: data/chunks
--chunk-size    Target maximum characters per chunk.    Default: from Settings
--overlap       Character overlap between chunks.       Default: from Settings
--threshold     Breakpoint threshold for semantic mode. Default: from Settings
--log-level     Logging verbosity.                      Default: INFO
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chunking import FixedSizeChunker, RecursiveStructureChunker, SemanticChunker
from app.chunking.base import BaseChunker, Chunk
from app.config import get_settings
from app.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_processed_documents(input_dir: Path) -> List[dict]:
    """Load all JSON document files from *input_dir*.

    Args:
        input_dir: Directory produced by ``scripts/ingest.py``.

    Returns:
        List of dicts with ``content`` and ``metadata`` keys.
    """
    docs = []
    for json_file in sorted(input_dir.glob("*.json")):
        try:
            payload = json.loads(json_file.read_text(encoding="utf-8"))
            docs.append(payload)
            logger.debug("Loaded: %s", json_file.name)
        except Exception as exc:
            logger.error("Failed to load %s: %s", json_file.name, exc)
    return docs


def build_chunker(
    strategy: str,
    chunk_size: int,
    overlap: int,
    threshold: Optional[float],
) -> BaseChunker:
    """Instantiate the requested chunker with the given parameters.

    Args:
        strategy: One of ``'fixed'``, ``'recursive'``, ``'semantic'``.
        chunk_size: Target maximum chunk size in characters.
        overlap: Character overlap between consecutive chunks.
        threshold: Breakpoint threshold for semantic chunking.

    Returns:
        Configured :class:`~app.chunking.base.BaseChunker` instance.
    """
    if strategy == "fixed":
        return FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    if strategy == "recursive":
        return RecursiveStructureChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    if strategy == "semantic":
        settings = get_settings()
        thr = threshold if threshold is not None else settings.semantic_breakpoint_threshold
        return SemanticChunker(
            breakpoint_threshold=thr,
            min_chunk_size=settings.semantic_min_chunk_size,
            max_chunk_size=settings.semantic_max_chunk_size,
        )
    raise ValueError(f"Unknown strategy: {strategy!r}")


def chunk_documents(
    docs: List[dict],
    chunker: BaseChunker,
    output_dir: Path,
) -> int:
    """Chunk all *docs* and write results to *output_dir*.

    Args:
        docs: Processed document dicts from :func:`load_processed_documents`.
        chunker: Configured chunker instance.
        output_dir: Directory where chunk JSON files are written.

    Returns:
        Total number of chunks written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    total_chunks = 0

    for doc in docs:
        content = doc.get("content", "")
        meta = doc.get("metadata", {})
        doc_id = meta.get("document_id", "unknown")

        if not content:
            logger.warning("Document '%s' has no content, skipping.", doc_id)
            continue

        try:
            chunks: List[Chunk] = chunker.chunk(
                text=content,
                document_id=doc_id,
                metadata={
                    "source_file": meta.get("source_file", ""),
                    "section_heading": meta.get("section_heading"),
                    "page_number": meta.get("page_number"),
                    "file_type": meta.get("file_type", ""),
                    "title": meta.get("title"),
                },
            )
        except Exception as exc:
            logger.error("Chunking failed for '%s': %s", doc_id, exc)
            continue

        for chunk in chunks:
            out_path = output_dir / f"{chunk.chunk_id.replace('#', '__')}.json"
            payload = chunk.model_dump()
            out_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            total_chunks += 1

        logger.info(
            "Chunked '%s' -> %d chunk(s) [strategy=%s]",
            doc_id,
            len(chunks),
            chunker.strategy_name,
        )

    return total_chunks


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Chunk processed documents using a selected strategy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["fixed", "recursive", "semantic"],
        required=True,
        help="Chunking strategy to apply.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed",
        help="Directory containing processed JSON documents.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/chunks",
        help="Directory where chunk JSON files will be written.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.default_chunk_size,
        help="Target maximum characters per chunk.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=settings.default_chunk_overlap,
        help="Character overlap between consecutive chunks.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Breakpoint similarity threshold for semantic chunking (0-1).",
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
    output_dir = Path(args.output)

    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    docs = load_processed_documents(input_dir)
    if not docs:
        logger.warning("No processed documents found in: %s", input_dir)
        sys.exit(0)

    chunker = build_chunker(
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        threshold=args.threshold,
    )

    logger.info(
        "Starting chunking: strategy=%s, chunk_size=%d, overlap=%d, input=%s -> output=%s",
        args.strategy,
        args.chunk_size,
        args.overlap,
        input_dir,
        output_dir,
    )

    total = chunk_documents(docs, chunker, output_dir)
    logger.info("Chunking complete: %d total chunk(s) written to %s", total, output_dir)


if __name__ == "__main__":
    main()
