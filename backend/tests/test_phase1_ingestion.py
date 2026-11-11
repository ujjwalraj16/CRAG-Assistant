"""Phase 1 tests: configuration, SQLite schema, document loaders, cleaning, and chunking."""

from pathlib import Path

import docx
import pymupdf
import pytest

from app.database.db import init_db
from app.database.models import (
    delete_document_record,
    get_chunks_by_document_id,
    get_document_by_hash,
    insert_chunks,
    insert_document,
    list_documents,
)
from app.ingestion.chunker import chunk_sections, split_text_with_overlap
from app.ingestion.document_loader import (
    EmptyDocumentError,
    UnsupportedFileError,
    compute_file_hash,
    load_document,
)
from app.ingestion.text_cleaner import clean_text


def test_clean_text_normalizes_whitespace_and_control_chars():
    raw = "Hello\x00   World\r\n\r\n\r\nSecond\t\tLine  "
    cleaned = clean_text(raw)
    assert cleaned == "Hello World\n\nSecond Line"


def test_pdf_extraction_preserves_page_numbers(tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Page one warranty details.")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Page two technical specifications.")
    doc.save(str(pdf_path))
    doc.close()

    sections = load_document(pdf_path)
    assert len(sections) == 2
    assert sections[0].page_number == 1
    assert "warranty details" in sections[0].text
    assert sections[1].page_number == 2
    assert "technical specifications" in sections[1].text


def test_docx_extraction_reads_paragraphs_and_tables(tmp_path: Path):
    docx_path = tmp_path / "policy.docx"
    document = docx.Document()
    document.add_paragraph("Return Policy Overview")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Item"
    table.cell(0, 1).text = "Window"
    table.cell(1, 0).text = "Laptops"
    table.cell(1, 1).text = "30 days"
    document.save(str(docx_path))

    sections = load_document(docx_path)
    assert len(sections) == 1
    assert "Return Policy Overview" in sections[0].text
    assert "Laptops | 30 days" in sections[0].text


def test_csv_extraction_preserves_columns_and_rows(tmp_path: Path):
    csv_path = tmp_path / "products.csv"
    csv_path.write_text(
        "Product,Price,Category\nAeroBook 14,$1199,Laptops\nPulseWatch,$249,Wearables\n",
        encoding="utf-8",
    )

    sections = load_document(csv_path)
    assert len(sections) == 1
    assert "Row 1: Product: AeroBook 14 | Price: $1199 | Category: Laptops" in sections[0].text
    assert "Row 2: Product: PulseWatch | Price: $249 | Category: Wearables" in sections[0].text


def test_empty_and_unsupported_files_raise_clear_errors(tmp_path: Path):
    empty_txt = tmp_path / "empty.txt"
    empty_txt.write_text("   \n\t  ", encoding="utf-8")
    with pytest.raises(EmptyDocumentError):
        load_document(empty_txt)

    unsupported = tmp_path / "archive.zip"
    unsupported.write_bytes(b"PK\x03\x04")
    with pytest.raises(UnsupportedFileError):
        load_document(unsupported)


def test_chunking_with_overlap_and_metadata(tmp_path: Path):
    txt_path = tmp_path / "guide.txt"
    long_text = " ".join(f"Sentence number {i} explains battery care." for i in range(40))
    txt_path.write_text(long_text, encoding="utf-8")

    sections = load_document(txt_path)
    chunks = chunk_sections(
        sections=sections,
        document_id="doc-123",
        document_name="guide.txt",
        category="Electronics",
        chunk_size=200,
        chunk_overlap=50,
    )
    assert len(chunks) > 1
    assert all(chunk.document_id == "doc-123" for chunk in chunks)
    assert all(chunk.category == "Electronics" for chunk in chunks)
    assert all(len(chunk.text) <= 200 for chunk in chunks)

    with pytest.raises(ValueError):
        split_text_with_overlap("hello world", chunk_size=100, chunk_overlap=100)


def test_sqlite_document_and_chunk_lifecycle(tmp_path: Path):
    db_path = tmp_path / "test_crag.db"
    init_db(db_path)

    sample_file = tmp_path / "faq.txt"
    sample_file.write_text("Support hours are 9am to 6pm.", encoding="utf-8")
    file_hash = compute_file_hash(sample_file)

    inserted = insert_document(
        document_id="doc-1",
        filename="faq.txt",
        file_hash=file_hash,
        file_path=str(sample_file),
        file_type="txt",
        category="Policies",
        chunk_count=1,
        file_size=sample_file.stat().st_size,
        db_path=db_path,
    )
    assert inserted["filename"] == "faq.txt"
    assert get_document_by_hash(file_hash, db_path=db_path) is not None

    stored_chunks = insert_chunks(
        [
            {
                "chunk_id": "doc-1_chunk_0",
                "document_id": "doc-1",
                "document_name": "faq.txt",
                "category": "Policies",
                "page_number": None,
                "chunk_index": 0,
                "text": "Support hours are 9am to 6pm.",
            }
        ],
        db_path=db_path,
    )
    assert len(stored_chunks) == 1
    assert isinstance(stored_chunks[0]["faiss_id"], int)

    assert len(list_documents(category="Policies", db_path=db_path)) == 1
    assert len(get_chunks_by_document_id("doc-1", db_path=db_path)) == 1

    assert delete_document_record("doc-1", db_path=db_path) is True
    assert get_chunks_by_document_id("doc-1", db_path=db_path) == []

