"""Phase 2 tests: embeddings, FAISS vector indexing, metadata filtering, deletion, and re-indexing."""

from pathlib import Path

import numpy as np
import pytest

from app.database.db import init_db
from app.database.models import get_chunks_by_document_id, list_documents
from app.ingestion.indexer import (
    DuplicateDocumentError,
    FaissVectorStore,
    delete_document_and_vectors,
    embed_texts,
    ingest_document,
    reindex_document,
)
from app.retrieval.retriever import retrieve_chunks


VOCAB = [
    "warranty",
    "battery",
    "laptop",
    "refund",
    "return",
    "smartphone",
    "camera",
    "banking",
    "interest",
    "healthcare",
    "policy",
    "charger",
]


def deterministic_embedder(texts: list[str]) -> np.ndarray:
    """Create deterministic keyword-based unit vectors for fast offline FAISS tests."""
    vectors = np.zeros((len(texts), len(VOCAB) + 1), dtype=np.float32)
    for row_idx, text in enumerate(texts):
        lowered = text.lower()
        for col_idx, term in enumerate(VOCAB):
            vectors[row_idx, col_idx] = float(lowered.count(term))
        # Small tie-breaker dimension so non-matching texts remain non-zero unit vectors.
        vectors[row_idx, -1] = 0.05
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


def test_ingest_and_retrieve_with_metadata_filter(tmp_path: Path):
    db_path = tmp_path / "test.db"
    index_dir = tmp_path / "faiss"
    init_db(db_path)

    laptop_doc = tmp_path / "laptop_guide.txt"
    laptop_doc.write_text(
        "The AeroBook laptop includes a 2-year battery warranty and fast USB-C charger.",
        encoding="utf-8",
    )
    bank_doc = tmp_path / "banking_faq.txt"
    bank_doc.write_text(
        "Savings accounts earn 4.5 percent interest under standard banking policy.",
        encoding="utf-8",
    )

    ingest_document(
        file_path=laptop_doc,
        category="Laptops",
        db_path=db_path,
        index_dir=index_dir,
        embed_fn=deterministic_embedder,
    )
    ingest_document(
        file_path=bank_doc,
        category="Banking",
        db_path=db_path,
        index_dir=index_dir,
        embed_fn=deterministic_embedder,
    )

    results = retrieve_chunks(
        query="laptop battery warranty",
        top_k=2,
        db_path=db_path,
        index_dir=index_dir,
        embed_fn=deterministic_embedder,
    )
    assert len(results) == 2
    assert results[0].document_name == "laptop_guide.txt"
    assert results[0].similarity_score > results[1].similarity_score

    filtered = retrieve_chunks(
        query="policy interest",
        top_k=5,
        category="Banking",
        db_path=db_path,
        index_dir=index_dir,
        embed_fn=deterministic_embedder,
    )
    assert len(filtered) == 1
    assert filtered[0].category == "Banking"
    assert filtered[0].document_name == "banking_faq.txt"


def test_duplicate_document_rejected(tmp_path: Path):
    db_path = tmp_path / "test.db"
    index_dir = tmp_path / "faiss"
    init_db(db_path)

    doc_path = tmp_path / "manual.txt"
    doc_path.write_text("Smartphone camera supports 4K recording.", encoding="utf-8")

    ingest_document(
        file_path=doc_path,
        category="Smartphones",
        db_path=db_path,
        index_dir=index_dir,
        embed_fn=deterministic_embedder,
    )
    with pytest.raises(DuplicateDocumentError):
        ingest_document(
            file_path=doc_path,
            category="Smartphones",
            db_path=db_path,
            index_dir=index_dir,
            embed_fn=deterministic_embedder,
        )


def test_delete_and_reindex_sync_faiss_and_sqlite(tmp_path: Path):
    db_path = tmp_path / "test.db"
    index_dir = tmp_path / "faiss"
    init_db(db_path)

    doc_path = tmp_path / "warranty.txt"
    doc_path.write_text(
        " ".join(f"Section {i}: laptop battery warranty and refund policy." for i in range(25)),
        encoding="utf-8",
    )

    record = ingest_document(
        file_path=doc_path,
        category="Laptops",
        chunk_size=300,
        chunk_overlap=50,
        db_path=db_path,
        index_dir=index_dir,
        embed_fn=deterministic_embedder,
    )
    doc_id = record["id"]
    initial_total = FaissVectorStore(index_dir=index_dir).total_vectors
    assert initial_total == record["chunk_count"]

    reindexed = reindex_document(
        document_id=doc_id,
        chunk_size=150,
        chunk_overlap=25,
        db_path=db_path,
        index_dir=index_dir,
        embed_fn=deterministic_embedder,
    )
    assert reindexed["new_chunks"] > reindexed["previous_chunks"]
    assert FaissVectorStore(index_dir=index_dir).total_vectors == reindexed["new_chunks"]

    deleted_count = delete_document_and_vectors(
        document_id=doc_id,
        db_path=db_path,
        index_dir=index_dir,
    )
    assert deleted_count == reindexed["new_chunks"]
    assert list_documents(db_path=db_path) == []
    assert get_chunks_by_document_id(doc_id, db_path=db_path) == []
    assert FaissVectorStore(index_dir=index_dir).total_vectors == 0


def test_empty_index_and_empty_query_return_empty_results(tmp_path: Path):
    db_path = tmp_path / "test.db"
    index_dir = tmp_path / "faiss"
    init_db(db_path)

    assert retrieve_chunks("any query", db_path=db_path, index_dir=index_dir, embed_fn=deterministic_embedder) == []
    assert retrieve_chunks("   ", db_path=db_path, index_dir=index_dir, embed_fn=deterministic_embedder) == []


def test_sentence_transformer_embedding_generation():
    vectors = embed_texts(
        [
            "Laptop battery replacement warranty policy",
            "How to bake sourdough bread at home",
        ]
    )
    assert vectors.shape[0] == 2
    assert vectors.shape[1] == 384
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)

