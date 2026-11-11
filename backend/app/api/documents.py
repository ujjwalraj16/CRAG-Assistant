"""Document upload, listing, search, deletion, and re-indexing API endpoints."""

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status

from app.config import get_public_settings
from app.database.models import get_all_setting_overrides, list_categories, list_documents
from app.ingestion.document_loader import (
    SUPPORTED_EXTENSIONS,
    EmptyDocumentError,
    UnsupportedFileError,
)
from app.ingestion.indexer import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    delete_document_and_vectors,
    ingest_document,
    reindex_document,
    save_uploaded_file_copy,
)
from app.schemas.documents import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentRecord,
    DocumentReindexResponse,
    DocumentUploadResponse,
)


router = APIRouter(prefix="/api/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    category: str = Form("General"),
) -> DocumentUploadResponse:
    """Upload, parse, chunk, embed, and index a PDF, DOCX, TXT, or CSV document."""
    original_name = Path(file.filename or "untitled").name
    extension = Path(original_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{extension}'. Supported formats: {supported}",
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file '{original_name}' is empty.",
        )

    saved_path = save_uploaded_file_copy(raw_bytes, original_name)
    runtime_cfg = get_public_settings(get_all_setting_overrides())

    try:
        doc_row = ingest_document(
            file_path=saved_path,
            filename=original_name,
            category=category,
            chunk_size=runtime_cfg["chunk_size"],
            chunk_overlap=runtime_cfg["chunk_overlap"],
            embedding_model=runtime_cfg["embedding_model"],
        )
    except DuplicateDocumentError as exc:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except (EmptyDocumentError, UnsupportedFileError, ValueError) as exc:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process and index '{original_name}': {exc}",
        ) from exc

    return DocumentUploadResponse(
        document=DocumentRecord(**doc_row),
        message=f"Successfully indexed '{original_name}' into {doc_row['chunk_count']} chunk(s).",
    )


@router.get("", response_model=DocumentListResponse)
def get_documents(
    search: str | None = Query(default=None),
    category: str | None = Query(default=None),
) -> DocumentListResponse:
    """List ingested documents with optional filename search and category filtering."""
    docs = list_documents(search=search, category=category)
    categories = list_categories()
    records = [DocumentRecord(**doc) for doc in docs]
    return DocumentListResponse(
        documents=records,
        categories=categories,
        total=len(records),
    )


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
def delete_document(document_id: str) -> DocumentDeleteResponse:
    """Delete a document from SQLite metadata, FAISS index, and disk storage."""
    try:
        deleted_chunks = delete_document_and_vectors(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return DocumentDeleteResponse(
        document_id=document_id,
        deleted_chunks=deleted_chunks,
        message=f"Deleted document and removed {deleted_chunks} chunk(s) from FAISS.",
    )


@router.post("/{document_id}/reindex", response_model=DocumentReindexResponse)
def reindex_existing_document(document_id: str) -> DocumentReindexResponse:
    """Re-extract, re-chunk, and re-embed an existing document using current settings."""
    runtime_cfg = get_public_settings(get_all_setting_overrides())
    try:
        result = reindex_document(
            document_id=document_id,
            chunk_size=runtime_cfg["chunk_size"],
            chunk_overlap=runtime_cfg["chunk_overlap"],
            embedding_model=runtime_cfg["embedding_model"],
        )
    except (DocumentNotFoundError, FileNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to re-index document: {exc}",
        ) from exc

    return DocumentReindexResponse(
        document=DocumentRecord(**result["document"]),
        previous_chunks=result["previous_chunks"],
        new_chunks=result["new_chunks"],
        message=(
            f"Re-indexed '{result['document']['filename']}': "
            f"{result['previous_chunks']} -> {result['new_chunks']} chunk(s)."
        ),
    )

