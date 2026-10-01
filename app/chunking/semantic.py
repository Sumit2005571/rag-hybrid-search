"""Semantic chunking using embedding cosine-similarity to detect topic shifts.

Algorithm
---------
1. Split the document into *sentences* (using simple regex).
2. For each pair of consecutive sentences, compute the cosine similarity of
   their embeddings using the configured embedding model.
3. When similarity drops below *breakpoint_threshold*, start a new chunk.
4. Consecutive sentences in the same chunk are joined together.
5. If no API key is configured, or if the embedding call fails, the chunker
   falls back to paragraph-boundary splitting (``\\n\\n``).

Configuration (all overridable via :class:`~app.config.Settings`)
-----------------------------------------------------------------
``SEMANTIC_BREAKPOINT_THRESHOLD``   float, default 0.75
    Cosine-similarity value below which a topic boundary is declared.
``SEMANTIC_MIN_CHUNK_SIZE``         int, default 100
    Minimum character length; chunks shorter than this are merged with their
    neighbour before output.
``SEMANTIC_MAX_CHUNK_SIZE``         int, default 2000
    Hard upper limit; chunks exceeding this are split at the last sentence
    boundary that fits.
``OPENAI_EMBEDDING_MODEL``          str, default ``"text-embedding-3-small"``
    Which OpenAI model to call for embeddings.
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from app.chunking.base import BaseChunker, Chunk, generate_chunk_id
from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

# Simple sentence splitter: split after ". " / "? " / "! " but keep the
# delimiter attached to the preceding sentence.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Return the cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _split_sentences(text: str) -> List[str]:
    """Split *text* into sentences, preserving content."""
    sentences = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _paragraph_fallback(text: str, chunk_size: int) -> List[str]:
    """Paragraph-based fallback when embeddings are unavailable.

    Splits on ``\\n\\n``, then hard-splits any paragraph that still exceeds
    *chunk_size*.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    result: List[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            result.append(para)
        else:
            # Hard split
            for i in range(0, len(para), chunk_size):
                result.append(para[i: i + chunk_size])
    return result or [text[:chunk_size]]


class SemanticChunker(BaseChunker):
    """Chunk text by detecting topic boundaries via embedding cosine-similarity.

    When an OpenAI API key is present the chunker embeds each sentence and
    groups consecutive sentences whose pairwise similarity stays above
    *breakpoint_threshold*.  When no key is configured or an API call fails,
    it transparently falls back to paragraph-boundary splitting.

    Args:
        breakpoint_threshold: Similarity below which a new chunk starts.
            Loaded from ``SEMANTIC_BREAKPOINT_THRESHOLD`` env var if not given.
        min_chunk_size: Chunks shorter than this are merged with a neighbour.
            Loaded from ``SEMANTIC_MIN_CHUNK_SIZE`` env var if not given.
        max_chunk_size: Hard limit; larger chunks are re-split at sentence
            boundaries.  Loaded from ``SEMANTIC_MAX_CHUNK_SIZE`` env var.
        embedding_fn: Optional callable ``(texts: List[str]) -> List[List[float]]``
            used for testing without hitting the OpenAI API.
    """

    strategy_name: str = "semantic"

    def __init__(
        self,
        breakpoint_threshold: Optional[float] = None,
        min_chunk_size: Optional[int] = None,
        max_chunk_size: Optional[int] = None,
        embedding_fn: Optional[Any] = None,
    ) -> None:
        settings = get_settings()
        self.breakpoint_threshold: float = (
            breakpoint_threshold
            if breakpoint_threshold is not None
            else settings.semantic_breakpoint_threshold
        )
        self.min_chunk_size: int = (
            min_chunk_size
            if min_chunk_size is not None
            else settings.semantic_min_chunk_size
        )
        self.max_chunk_size: int = (
            max_chunk_size
            if max_chunk_size is not None
            else settings.semantic_max_chunk_size
        )
        # Allow injecting a test double
        self._embedding_fn = embedding_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        text: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Split *text* at semantic topic boundaries.

        Args:
            text: Document text to split.
            document_id: Deterministic parent document ID.
            metadata: Document-level metadata propagated to each chunk.

        Returns:
            Ordered list of :class:`~app.chunking.base.Chunk` objects.
        """
        if not text:
            return []

        meta = metadata or {}
        source_file = meta.get("source_file", "")
        section_heading = meta.get("section_heading")
        page_number = meta.get("page_number")
        extra_meta = {k: v for k, v in meta.items()
                      if k not in ("source_file", "section_heading", "page_number")}

        raw_segments = self._build_segments(text)

        chunks: List[Chunk] = []
        char_offset = 0

        for idx, segment in enumerate(raw_segments):
            start = text.find(segment, char_offset)
            if start == -1:
                start = char_offset
            end = start + len(segment)
            char_offset = end

            chunks.append(
                Chunk(
                    chunk_id=generate_chunk_id(document_id, idx, self.strategy_name),
                    document_id=document_id,
                    source_file=source_file,
                    chunk_index=idx,
                    start_char=start,
                    end_char=end,
                    text=segment,
                    char_count=len(segment),
                    section_heading=section_heading,
                    page_number=page_number,
                    strategy=self.strategy_name,
                    metadata=extra_meta,
                )
            )

        logger.debug(
            "SemanticChunker produced %d chunks for document '%s'",
            len(chunks),
            document_id,
        )
        return chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_segments(self, text: str) -> List[str]:
        """Return a list of text segments based on semantic similarity.

        Falls back to paragraph splitting if embedding is unavailable.
        """
        sentences = _split_sentences(text)
        if not sentences:
            return [text] if text else []

        # Single sentence — no boundary detection needed
        if len(sentences) == 1:
            return sentences

        embeddings = self._get_embeddings(sentences)
        if embeddings is None:
            logger.warning(
                "Embedding call failed or API key absent — falling back to "
                "paragraph-boundary splitting."
            )
            return _paragraph_fallback(text, self.max_chunk_size)

        # Detect boundaries
        boundaries: List[int] = []  # indices of sentences that START a new chunk
        for i in range(len(sentences) - 1):
            sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
            if sim < self.breakpoint_threshold:
                boundaries.append(i + 1)

        # Build groups
        groups: List[List[str]] = []
        current_group: List[str] = []
        for i, sentence in enumerate(sentences):
            if i in boundaries and current_group:
                groups.append(current_group)
                current_group = [sentence]
            else:
                current_group.append(sentence)
        if current_group:
            groups.append(current_group)

        # Join each group into a segment string
        segments = [" ".join(g) for g in groups]

        # Enforce min / max size
        segments = self._enforce_min_size(segments)
        segments = self._enforce_max_size(segments)

        return segments

    def _get_embeddings(
        self, sentences: List[str]
    ) -> Optional[List[List[float]]]:
        """Return embeddings for *sentences* or ``None`` on failure."""
        if self._embedding_fn is not None:
            try:
                return self._embedding_fn(sentences)
            except Exception as exc:
                logger.error("Injected embedding_fn raised: %s", exc)
                return None

        # Use OpenAI via the project's embedding wrapper
        settings = get_settings()
        if not settings.openai_api_key:
            logger.debug("OPENAI_API_KEY not set; skipping embedding call.")
            return None

        try:
            from app.embeddings.openai_embeddings import OpenAIEmbeddingGenerator

            generator = OpenAIEmbeddingGenerator()
            return generator.embed_texts(sentences)
        except Exception as exc:
            logger.error("Embedding API call failed: %s", exc)
            return None

    def _enforce_min_size(self, segments: List[str]) -> List[str]:
        """Merge segments shorter than *min_chunk_size* with their neighbour."""
        if not segments or self.min_chunk_size <= 0:
            return segments

        result: List[str] = []
        buffer = ""
        for seg in segments:
            if len(seg) < self.min_chunk_size:
                buffer = (buffer + " " + seg).strip() if buffer else seg
            else:
                if buffer:
                    result.append(buffer)
                    buffer = ""
                result.append(seg)
        if buffer:
            if result:
                result[-1] = (result[-1] + " " + buffer).strip()
            else:
                result.append(buffer)
        return result

    def _enforce_max_size(self, segments: List[str]) -> List[str]:
        """Split segments that exceed *max_chunk_size* at sentence boundaries."""
        result: List[str] = []
        for seg in segments:
            if len(seg) <= self.max_chunk_size:
                result.append(seg)
            else:
                # Re-split this over-sized segment at sentence boundaries
                sub_sents = _split_sentences(seg)
                sub_buf = ""
                for s in sub_sents:
                    candidate = (sub_buf + " " + s).strip() if sub_buf else s
                    if len(candidate) <= self.max_chunk_size:
                        sub_buf = candidate
                    else:
                        if sub_buf:
                            result.append(sub_buf)
                        # If even one sentence is too long, hard-split it
                        if len(s) > self.max_chunk_size:
                            for i in range(0, len(s), self.max_chunk_size):
                                result.append(s[i: i + self.max_chunk_size])
                            sub_buf = ""
                        else:
                            sub_buf = s
                if sub_buf:
                    result.append(sub_buf)
        return result
