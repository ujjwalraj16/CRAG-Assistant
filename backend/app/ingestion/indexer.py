"""Embedding generation, FAISS vector store management, and document indexing."""

from pathlib import Path
import shutil
from typing import Any, Callable
import uuid

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.database.models import (
    delete_chunks_by_document_id,
    delete_document_record,
    get_all_chunks,
    get_document_by_hash,
    get_document_by_id,
    insert_chunks,
    insert_document,
    update_document_chunk_count,
)
from app.ingestion.chunker import chunk_sections
from app.ingestion.document_loader import compute_file_hash, load_document


EmbedFn = Callable[[list[str]], np.ndarray]

_MODEL_CACHE: dict[str, SentenceTransformer] = {}


class DuplicateDocumentError(ValueError):
    """Raised when an uploaded file matches an already-indexed document."""


class DocumentNotFoundError(LookupError):
    """Raised when a target document ID does not exist."""


def get_embedding_model(model_name: str | None = None) -> SentenceTransformer:
    """Load and cache the SentenceTransformer model in memory."""
    target_model = model_name or settings.embedding_model
    if target_model not in _MODEL_CACHE:
        _MODEL_CACHE[target_model] = SentenceTransformer(target_model)
    return _MODEL_CACHE[target_model]


def embed_texts(
    texts: list[str],
    model_name: str | None = None,
) -> np.ndarray:
    """Generate L2-normalized float32 embeddings so inner product equals cosine similarity."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    model = get_embedding_model(model_name)
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vectors_f32 = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors_f32.ndim == 1:
        vectors_f32 = vectors_f32.reshape(1, -1)
    # Normalize explicitly in case a custom model ignores normalize_embeddings.
    faiss.normalize_L2(vectors_f32)
    return vectors_f32


class FaissVectorStore:
    """Manages a persistent FAISS IndexIDMap2(IndexFlatIP) mapped to SQLite chunk IDs."""

    def __init__(self, index_dir: Path | None = None) -> None:
        self.index_dir = index_dir or settings.resolved_faiss_index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "vectors.faiss"
        self.index: faiss.Index | None = self._load_if_exists()

    def _load_if_exists(self) -> faiss.Index | None:
        if self.index_path.exists():
            return faiss.read_index(str(self.index_path))
        return None

    def _ensure_index(self, dimension: int) -> faiss.Index:
        if self.index is None:
            base_index = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIDMap2(base_index)
        elif self.index.d != dimension:
            raise ValueError(
                f"Embedding dimension mismatch: FAISS index expects {self.index.d}, "
                f"but received {dimension}. Re-index documents to switch embedding models."
            )
        return self.index

    @property
    def total_vectors(self) -> int:
        return int(self.index.ntotal) if self.index is not None else 0

    def save(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))

    def add_vectors(self, vectors: np.ndarray, faiss_ids: list[int]) -> None:
        """Add normalized vectors paired with their SQLite faiss_id integers."""
        if len(faiss_ids) == 0 or vectors.size == 0:
            return
        if vectors.shape[0] != len(faiss_ids):
            raise ValueError("Number of vectors must match number of faiss_ids.")

        vectors_f32 = np.ascontiguousarray(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors_f32)
        ids_i64 = np.ascontiguousarray(faiss_ids, dtype=np.int64)

        index = self._ensure_index(vectors_f32.shape[1])
        index.add_with_ids(vectors_f32, ids_i64)
        self.save()

    def remove_vectors(self, faiss_ids: list[int]) -> int:
        """Remove vectors by their integer IDs and persist the updated index."""
        if self.index is None or self.index.ntotal == 0 or not faiss_ids:
            return 0

        ids_i64 = np.ascontiguousarray(faiss_ids, dtype=np.int64)
        try:
            removed = int(self.index.remove_ids(ids_i64))
        except TypeError:
            selector = faiss.IDSelectorBatch(ids_i64.size, faiss.swig_ptr(ids_i64))
            removed = int(self.index.remove_ids(selector))

        self.save()
        return removed

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Search the index and return (faiss_id, cosine_similarity) pairs."""
        if self.index is None or self.index.ntotal == 0 or top_k <= 0:
            return []

        query_f32 = np.ascontiguousarray(query_vector, dtype=np.float32)
        if query_f32.ndim == 1:
            query_f32 = query_f32.reshape(1, -1)
        if query_f32.shape[1] != self.index.d:
            raise ValueError(
                f"Query embedding dimension ({query_f32.shape[1]}) does not match "
                f"FAISS index dimension ({self.index.d})."
            )
        faiss.normalize_L2(query_f32)

        k = min(top_k, int(self.index.ntotal))
        scores, ids = self.index.search(query_f32, k)

        matches: list[tuple[int, float]] = []
        for faiss_id, score in zip(ids[0], scores[0]):
            if int(faiss_id) != -1:
                matches.append((int(faiss_id), float(score)))
        return matches

    def reset(self) -> None:
        """Clear the in-memory and on-disk FAISS index."""
        self.index = None
        if self.index_path.exists():
            self.index_path.unlink()


def ingest_document(
    file_path: Path,
    filename: str | None = None,
    category: str = "General",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    embedding_model: str | None = None,
    db_path: Path | None = None,
    index_dir: Path | None = None,
    embed_fn: EmbedFn | None = None,
) -> dict[str, Any]:
    """Load, clean, chunk, embed, and index a file into SQLite and FAISS."""
    effective_name = filename or file_path.name
    effective_category = category.strip() or "General"
    effective_chunk_size = chunk_size or settings.chunk_size
    effective_chunk_overlap = (
        settings.chunk_overlap if chunk_overlap is None else chunk_overlap
    )

    file_hash = compute_file_hash(file_path)
    existing = get_document_by_hash(file_hash, db_path=db_path)
    if existing is not None:
        raise DuplicateDocumentError(
            f"Document '{effective_name}' duplicates already-indexed '{existing['filename']}' "
            f"(id={existing['id']})."
        )

    sections = load_document(file_path)
    document_id = str(uuid.uuid4())
    chunks = chunk_sections(
        sections=sections,
        document_id=document_id,
        document_name=effective_name,
        category=effective_category,
        chunk_size=effective_chunk_size,
        chunk_overlap=effective_chunk_overlap,
    )

    doc_record = insert_document(
        document_id=document_id,
        filename=effective_name,
        file_hash=file_hash,
        file_path=str(file_path),
        file_type=file_path.suffix.lower().lstrip("."),
        category=effective_category,
        chunk_count=len(chunks),
        file_size=file_path.stat().st_size,
        db_path=db_path,
    )

    try:
        stored_chunks = insert_chunks(
            [chunk.model_dump() for chunk in chunks],
            db_path=db_path,
        )
        texts = [chunk["text"] for chunk in stored_chunks]
        faiss_ids = [int(chunk["faiss_id"]) for chunk in stored_chunks]

        vectors = (
            embed_fn(texts)
            if embed_fn is not None
            else embed_texts(texts, model_name=embedding_model)
        )
        vector_store = FaissVectorStore(index_dir=index_dir)
        vector_store.add_vectors(vectors, faiss_ids)
    except Exception:
        # Roll back SQLite document/chunks if embedding or FAISS persistence fails.
        delete_document_record(document_id, db_path=db_path)
        raise

    return doc_record


def delete_document_and_vectors(
    document_id: str,
    db_path: Path | None = None,
    index_dir: Path | None = None,
    upload_dir: Path | None = None,
) -> int:
    """Delete a document from SQLite, FAISS, and upload storage. Returns deleted chunk count."""
    doc = get_document_by_id(document_id, db_path=db_path)
    if doc is None:
        raise DocumentNotFoundError(f"Document '{document_id}' was not found.")

    faiss_ids = delete_chunks_by_document_id(document_id, db_path=db_path)
    vector_store = FaissVectorStore(index_dir=index_dir)
    vector_store.remove_vectors(faiss_ids)

    delete_document_record(document_id, db_path=db_path)

    # Only delete the source file from disk if it resides in the managed upload directory.
    managed_upload_dir = (upload_dir or settings.resolved_upload_dir).resolve()
    stored_path = Path(doc["file_path"]).resolve()
    if stored_path.exists() and managed_upload_dir in stored_path.parents:
        stored_path.unlink()

    return len(faiss_ids)


def reindex_document(
    document_id: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    embedding_model: str | None = None,
    db_path: Path | None = None,
    index_dir: Path | None = None,
    embed_fn: EmbedFn | None = None,
) -> dict[str, Any]:
    """Re-extract, re-chunk, and re-embed an existing document with current settings."""
    doc = get_document_by_id(document_id, db_path=db_path)
    if doc is None:
        raise DocumentNotFoundError(f"Document '{document_id}' was not found.")

    source_path = Path(doc["file_path"])
    if not source_path.exists():
        raise FileNotFoundError(
            f"Original uploaded file for '{doc['filename']}' is missing at {source_path}."
        )

    effective_chunk_size = chunk_size or settings.chunk_size
    effective_chunk_overlap = (
        settings.chunk_overlap if chunk_overlap is None else chunk_overlap
    )
    previous_chunks = int(doc["chunk_count"])

    sections = load_document(source_path)
    new_chunks = chunk_sections(
        sections=sections,
        document_id=document_id,
        document_name=doc["filename"],
        category=doc["category"],
        chunk_size=effective_chunk_size,
        chunk_overlap=effective_chunk_overlap,
    )
    texts = [chunk.text for chunk in new_chunks]
    vectors = (
        embed_fn(texts)
        if embed_fn is not None
        else embed_texts(texts, model_name=embedding_model)
    )

    old_faiss_ids = delete_chunks_by_document_id(document_id, db_path=db_path)
    vector_store = FaissVectorStore(index_dir=index_dir)
    vector_store.remove_vectors(old_faiss_ids)

    stored_chunks = insert_chunks(
        [chunk.model_dump() for chunk in new_chunks],
        db_path=db_path,
    )
    new_faiss_ids = [int(chunk["faiss_id"]) for chunk in stored_chunks]
    vector_store.add_vectors(vectors, new_faiss_ids)

    updated_doc = update_document_chunk_count(document_id, len(stored_chunks), db_path=db_path)
    assert updated_doc is not None
    return {
        "document": updated_doc,
        "previous_chunks": previous_chunks,
        "new_chunks": len(stored_chunks),
    }


def rebuild_faiss_index_from_db(
    embedding_model: str | None = None,
    db_path: Path | None = None,
    index_dir: Path | None = None,
    embed_fn: EmbedFn | None = None,
) -> int:
    """Rebuild the entire FAISS index from chunks currently stored in SQLite."""
    chunks = get_all_chunks(db_path=db_path)
    vector_store = FaissVectorStore(index_dir=index_dir)
    vector_store.reset()

    if not chunks:
        return 0

    texts = [chunk["text"] for chunk in chunks]
    faiss_ids = [int(chunk["faiss_id"]) for chunk in chunks]
    vectors = (
        embed_fn(texts)
        if embed_fn is not None
        else embed_texts(texts, model_name=embedding_model)
    )
    vector_store.add_vectors(vectors, faiss_ids)
    return len(chunks)


def save_uploaded_file_copy(
    source_bytes: bytes,
    original_filename: str,
    upload_dir: Path | None = None,
) -> Path:
    """Persist an uploaded file safely inside the managed upload directory."""
    target_dir = upload_dir or settings.resolved_upload_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(original_filename).name
    target_path = target_dir / f"{uuid.uuid4().hex}_{safe_name}"
    target_path.write_bytes(source_bytes)
    return target_path


def copy_file_to_uploads(source_path: Path, upload_dir: Path | None = None) -> Path:
    """Copy a local file into the managed upload directory."""
    target_dir = upload_dir or settings.resolved_upload_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{uuid.uuid4().hex}_{source_path.name}"
    shutil.copy2(source_path, target_path)
    return target_path
