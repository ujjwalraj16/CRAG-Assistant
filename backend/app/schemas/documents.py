"""Schemas for document ingestion, listing, deletion, and re-indexing."""

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents a single text chunk with source metadata."""

    chunk_id: str
    document_id: str
    document_name: str
    category: str
    page_number: int | None = None
    chunk_index: int
    text: str
    faiss_id: int | None = None


class DocumentRecord(BaseModel):
    """Stored metadata for an ingested document."""

    id: str
    filename: str
    file_type: str
    category: str
    chunk_count: int
    file_size: int
    created_at: str
    updated_at: str


class DocumentUploadResponse(BaseModel):
    """Response returned after successful document upload and indexing."""

    document: DocumentRecord
    message: str


class DocumentListResponse(BaseModel):
    """Response containing filtered documents and available categories."""

    documents: list[DocumentRecord] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    total: int


class DocumentDeleteResponse(BaseModel):
    """Response returned when a document is deleted from storage and index."""

    document_id: str
    deleted_chunks: int
    message: str


class DocumentReindexResponse(BaseModel):
    """Response returned after re-chunking and re-embedding an existing document."""

    document: DocumentRecord
    previous_chunks: int
    new_chunks: int
    message: str

