"""Internal vector similarity retrieval over the FAISS index and SQLite metadata."""

from pathlib import Path

from app.config import settings
from app.database.models import get_chunks_by_faiss_ids
from app.ingestion.indexer import EmbedFn, FaissVectorStore, embed_texts
from app.schemas.pipeline import RetrievedChunk


def retrieve_chunks(
    query: str,
    top_k: int | None = None,
    category: str | None = None,
    document_ids: list[str] | None = None,
    min_similarity: float | None = None,
    embedding_model: str | None = None,
    db_path: Path | None = None,
    index_dir: Path | None = None,
    embed_fn: EmbedFn | None = None,
) -> list[RetrievedChunk]:
    """Embed a query, search FAISS, apply optional metadata filters, and return top-k chunks."""
    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    k = top_k or settings.retrieval_top_k
    vector_store = FaissVectorStore(index_dir=index_dir)
    if vector_store.total_vectors == 0:
        return []

    query_vector = (
        embed_fn([cleaned_query])
        if embed_fn is not None
        else embed_texts([cleaned_query], model_name=embedding_model)
    )

    # Fetch a broader candidate set when filtering by metadata so top_k stays populated.
    has_metadata_filter = bool(
        (category and category.strip() and category.strip().lower() != "all")
        or document_ids
    )
    search_k = (
        min(vector_store.total_vectors, max(k * 6, 50))
        if has_metadata_filter
        else min(vector_store.total_vectors, k)
    )

    raw_matches = vector_store.search(query_vector, top_k=search_k)
    if not raw_matches:
        return []

    chunk_map = get_chunks_by_faiss_ids(
        [faiss_id for faiss_id, _ in raw_matches],
        db_path=db_path,
    )

    normalized_category = (
        category.strip().lower()
        if category and category.strip() and category.strip().lower() != "all"
        else None
    )
    allowed_doc_ids = set(document_ids) if document_ids else None
    threshold = 0.0 if min_similarity is None else float(min_similarity)

    results: list[RetrievedChunk] = []
    for faiss_id, similarity in raw_matches:
        chunk_row = chunk_map.get(faiss_id)
        if chunk_row is None:
            continue
        if similarity < threshold:
            continue
        if normalized_category and chunk_row["category"].strip().lower() != normalized_category:
            continue
        if allowed_doc_ids and chunk_row["document_id"] not in allowed_doc_ids:
            continue

        results.append(
            RetrievedChunk(
                chunk_id=chunk_row["chunk_id"],
                document_id=chunk_row["document_id"],
                document_name=chunk_row["document_name"],
                category=chunk_row["category"],
                page_number=chunk_row["page_number"],
                chunk_index=chunk_row["chunk_index"],
                text=chunk_row["text"],
                similarity_score=round(float(similarity), 4),
                faiss_id=faiss_id,
            )
        )
        if len(results) >= k:
            break

    return results

