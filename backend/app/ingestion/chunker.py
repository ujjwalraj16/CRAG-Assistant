"""Overlapping text chunker that preserves document and page metadata."""

from app.ingestion.document_loader import ExtractedSection
from app.schemas.documents import DocumentChunk


BOUNDARY_SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", " ")


def _find_split_end(text: str, start: int, max_end: int) -> int:
    """Choose a clean boundary near max_end so chunks avoid splitting words or rows mid-token."""
    if max_end >= len(text):
        return len(text)

    window_start = start + (max_end - start) // 2
    window = text[window_start:max_end]

    for sep in BOUNDARY_SEPARATORS:
        pos = window.rfind(sep)
        if pos != -1:
            candidate = window_start + pos + len(sep)
            if candidate > start:
                return candidate
    return max_end


def split_text_with_overlap(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[str]:
    """Split a text string into overlapping segments of up to chunk_size characters."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be strictly smaller than chunk_size.")

    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    text_length = len(cleaned)

    while start < text_length:
        max_end = min(start + chunk_size, text_length)
        end = _find_split_end(cleaned, start, max_end)

        segment = cleaned[start:end].strip()
        if segment:
            chunks.append(segment)

        if end >= text_length:
            break

        # Ensure start strictly advances so unbreakable strings cannot cause an infinite loop.
        next_start = max(end - chunk_overlap, start + 1)
        start = next_start

    return chunks


def chunk_sections(
    sections: list[ExtractedSection],
    document_id: str,
    document_name: str,
    category: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[DocumentChunk]:
    """Convert extracted document sections into metadata-rich DocumentChunk objects."""
    document_chunks: list[DocumentChunk] = []
    chunk_index = 0

    for section in sections:
        pieces = split_text_with_overlap(
            section.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for piece in pieces:
            document_chunks.append(
                DocumentChunk(
                    chunk_id=f"{document_id}_chunk_{chunk_index}",
                    document_id=document_id,
                    document_name=document_name,
                    category=category,
                    page_number=section.page_number,
                    chunk_index=chunk_index,
                    text=piece,
                )
            )
            chunk_index += 1

    return document_chunks

