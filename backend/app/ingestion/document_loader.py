"""Document loaders for PDF, DOCX, TXT, and CSV files."""

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable

import docx
import pymupdf

from app.ingestion.text_cleaner import clean_text


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv"}


class UnsupportedFileError(ValueError):
    """Raised when a file extension is not supported by the ingestion pipeline."""


class EmptyDocumentError(ValueError):
    """Raised when a file contains no extractable text."""


@dataclass
class ExtractedSection:
    """Represents a cleaned block of text extracted from a file or page."""

    text: str
    page_number: int | None = None


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file to prevent duplicate ingestion."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as file_handle:
        for block in iter(lambda: file_handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pdf(file_path: Path) -> list[ExtractedSection]:
    """Extract cleaned text from each non-empty PDF page while preserving page numbers."""
    sections: list[ExtractedSection] = []
    with pymupdf.open(str(file_path)) as pdf_doc:
        for page_idx, page in enumerate(pdf_doc):
            cleaned = clean_text(page.get_text("text"))
            if cleaned:
                sections.append(ExtractedSection(text=cleaned, page_number=page_idx + 1))
    return sections


def _load_docx(file_path: Path) -> list[ExtractedSection]:
    """Extract cleaned paragraphs and table rows from a DOCX file."""
    doc = docx.Document(str(file_path))
    parts: list[str] = []

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            parts.append(text)

    for table in doc.tables:
        for row in table.rows:
            cells = [clean_text(cell.text) for cell in row.cells]
            non_empty = [cell for cell in cells if cell]
            if non_empty:
                parts.append(" | ".join(non_empty))

    combined = clean_text("\n\n".join(parts))
    return [ExtractedSection(text=combined, page_number=None)] if combined else []


def _load_txt(file_path: Path) -> list[ExtractedSection]:
    """Extract text from a plain text file with UTF-8/Latin-1 decoding fallback."""
    raw_bytes = file_path.read_bytes()
    try:
        decoded = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        decoded = raw_bytes.decode("latin-1")

    cleaned = clean_text(decoded)
    return [ExtractedSection(text=cleaned, page_number=None)] if cleaned else []


def _load_csv(file_path: Path) -> list[ExtractedSection]:
    """Convert CSV rows into searchable text while preserving column names and row context."""
    raw_bytes = file_path.read_bytes()
    try:
        decoded = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        decoded = raw_bytes.decode("latin-1")

    lines = decoded.splitlines()
    if not lines:
        return []

    reader = csv.DictReader(lines)
    if not reader.fieldnames:
        return []

    row_lines: list[str] = []
    for row_index, row in enumerate(reader, start=1):
        pairs: list[str] = []
        for col_name, value in row.items():
            if col_name is None:
                continue
            cleaned_col = clean_text(str(col_name))
            cleaned_val = clean_text(str(value)) if value is not None else ""
            if cleaned_col and cleaned_val:
                pairs.append(f"{cleaned_col}: {cleaned_val}")
        if pairs:
            row_lines.append(f"Row {row_index}: " + " | ".join(pairs))

    combined = clean_text("\n".join(row_lines))
    return [ExtractedSection(text=combined, page_number=None)] if combined else []


LOADERS: dict[str, Callable[[Path], list[ExtractedSection]]] = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".txt": _load_txt,
    ".csv": _load_csv,
}


def load_document(file_path: Path) -> list[ExtractedSection]:
    """Load and clean text sections from a supported file, raising clear errors on invalid input."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = file_path.suffix.lower()
    loader = LOADERS.get(extension)
    if loader is None:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileError(
            f"Unsupported file format '{extension}'. Supported formats: {supported}"
        )

    sections = loader(file_path)
    if not sections:
        raise EmptyDocumentError(f"Document '{file_path.name}' contains no extractable text.")
    return sections

