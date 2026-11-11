"""Evidence filtering and deduplication across internal documents and external web results."""

import re

from app.schemas.pipeline import (
    EvidenceFilterResult,
    GradedChunk,
    RejectedEvidence,
)


WHITESPACE_RE = re.compile(r"\s+")
DEFAULT_MIN_CONFIDENCE = 0.45


def _text_fingerprint(text: str) -> str:
    """Create a normalized text key to detect duplicate evidence across chunks or URLs."""
    return WHITESPACE_RE.sub(" ", text.strip().lower())


def filter_evidence(
    internal_chunks: list[GradedChunk],
    web_chunks: list[GradedChunk] | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> EvidenceFilterResult:
    """
    Combine internal and external graded chunks, keeping only high-confidence unique evidence.

    Preserves document references, page numbers, and web URLs on accepted items and tracks
    explicit rejection reasons for pipeline inspection.
    """
    candidates = list(internal_chunks) + list(web_chunks or [])

    # Sort highest-confidence and internal sources first so deduplication retains the best copy.
    candidates.sort(
        key=lambda item: (
            1 if item.relevance == "relevant" else 0,
            round(item.confidence, 4),
            1 if item.source_type == "internal" else 0,
            round(item.similarity_score, 4),
        ),
        reverse=True,
    )

    accepted: list[GradedChunk] = []
    rejected: list[RejectedEvidence] = []
    seen_chunk_ids: set[str] = set()
    seen_text_keys: set[str] = set()
    seen_urls: set[str] = set()

    for chunk in candidates:
        if chunk.relevance != "relevant":
            rejected.append(
                RejectedEvidence(
                    chunk=chunk,
                    rejection_reason=f"Irrelevant to query ({chunk.reason})",
                )
            )
            continue

        if chunk.confidence < min_confidence:
            rejected.append(
                RejectedEvidence(
                    chunk=chunk,
                    rejection_reason=(
                        f"Low relevance confidence ({chunk.confidence:.2f} < {min_confidence:.2f})"
                    ),
                )
            )
            continue

        text_key = _text_fingerprint(chunk.text)
        if not text_key:
            rejected.append(
                RejectedEvidence(chunk=chunk, rejection_reason="Empty evidence text")
            )
            continue

        if chunk.chunk_id in seen_chunk_ids or text_key in seen_text_keys:
            rejected.append(
                RejectedEvidence(chunk=chunk, rejection_reason="Duplicate evidence content")
            )
            continue

        if chunk.source_type == "web" and chunk.url:
            normalized_url = chunk.url.strip().rstrip("/").lower()
            if normalized_url in seen_urls:
                rejected.append(
                    RejectedEvidence(chunk=chunk, rejection_reason="Duplicate web source URL")
                )
                continue
            seen_urls.add(normalized_url)

        seen_chunk_ids.add(chunk.chunk_id)
        seen_text_keys.add(text_key)
        accepted.append(chunk)

    internal_accepted = sum(1 for item in accepted if item.source_type == "internal")
    web_accepted = sum(1 for item in accepted if item.source_type == "web")

    return EvidenceFilterResult(
        accepted=accepted,
        rejected=rejected,
        internal_accepted_count=internal_accepted,
        web_accepted_count=web_accepted,
        rejected_count=len(rejected),
    )

