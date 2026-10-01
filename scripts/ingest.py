"""CLI script for batch document ingestion.

Usage:
    python scripts/ingest.py --input data/raw --output data/processed

Scans the input directory for supported files, loads and normalizes them,
then writes processed JSON documents to the output directory.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Ensure project root is on the path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.loaders import DocumentLoader
from app.ingestion.normalizer import DocumentNormalizer
from app.ingestion.metadata import RawDocument
from app.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def ingest_directory(
    input_dir: Path,
    output_dir: Path,
    loader: DocumentLoader,
    normalizer: DocumentNormalizer,
) -> List[Path]:
    """Ingest all supported files from *input_dir* into *output_dir*.

    Each document is saved as a ``.json`` file containing the normalized text
    and full metadata.  Filenames are based on the deterministic document ID
    so re-running the same ingestion is idempotent.

    Args:
        input_dir: Directory containing raw source documents.
        output_dir: Directory where processed JSON files will be written.
        loader: Configured ``DocumentLoader`` instance.
        normalizer: Configured ``DocumentNormalizer`` instance.

    Returns:
        List of paths to the processed output files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    supported = set(loader.supported_extensions)
    output_files: List[Path] = []

    files = sorted(input_dir.iterdir())
    if not files:
        logger.warning("Input directory is empty: %s", input_dir)
        return output_files

    for file_path in files:
        if file_path.is_dir():
            continue
        if file_path.suffix.lower() not in supported:
            logger.debug("Skipping unsupported file: %s", file_path.name)
            continue

        try:
            raw_docs = loader.load(file_path)
        except Exception as exc:
            logger.error("Failed to load %s: %s", file_path.name, exc)
            continue

        for raw_doc in raw_docs:
            try:
                normalized = normalizer.normalize(raw_doc)
            except Exception as exc:
                logger.error(
                    "Failed to normalize document %s from %s: %s",
                    raw_doc.metadata.document_id,
                    file_path.name,
                    exc,
                )
                continue

            out_name = f"{normalized.metadata.document_id}.json"
            out_path = output_dir / out_name
            payload = {
                "content": normalized.content,
                "metadata": normalized.metadata.model_dump(),
            }
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            output_files.append(out_path)
            logger.info("Processed -> %s", out_path.name)

    return output_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest raw documents into normalized JSON format."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw",
        help="Path to directory containing raw documents (default: data/raw)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Path to directory for processed documents (default: data/processed)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    loader = DocumentLoader()
    normalizer = DocumentNormalizer()

    logger.info("Starting ingestion: %s -> %s", input_dir, output_dir)
    outputs = ingest_directory(input_dir, output_dir, loader, normalizer)
    logger.info("Ingestion complete: %d document(s) processed.", len(outputs))


if __name__ == "__main__":
    main()
