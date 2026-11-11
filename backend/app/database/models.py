"""Straightforward SQLite data access functions for documents, chunks, traces, and settings."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from app.database.db import get_connection


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Document Operations
# ---------------------------------------------------------------------------

def insert_document(
    document_id: str,
    filename: str,
    file_hash: str,
    file_path: str,
    file_type: str,
    category: str,
    chunk_count: int,
    file_size: int,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Insert a new document record and return it as a dictionary."""
    now = utc_now_iso()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO documents (
                id, filename, file_hash, file_path, file_type,
                category, chunk_count, file_size, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                filename,
                file_hash,
                file_path,
                file_type,
                category,
                chunk_count,
                file_size,
                now,
                now,
            ),
        )
    doc = get_document_by_id(document_id, db_path=db_path)
    assert doc is not None
    return doc


def get_document_by_id(document_id: str, db_path: Path | None = None) -> dict[str, Any] | None:
    """Fetch a document record by its UUID."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
    return dict(row) if row else None


def get_document_by_hash(file_hash: str, db_path: Path | None = None) -> dict[str, Any] | None:
    """Fetch an existing document with the same SHA-256 hash to prevent duplicates."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE file_hash = ?",
            (file_hash,),
        ).fetchone()
    return dict(row) if row else None


def list_documents(
    search: str | None = None,
    category: str | None = None,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """List stored documents with optional filename search and category metadata filter."""
    query = "SELECT * FROM documents WHERE 1=1"
    params: list[Any] = []

    if search and search.strip():
        query += " AND (LOWER(filename) LIKE ? OR LOWER(category) LIKE ?)"
        term = f"%{search.strip().lower()}%"
        params.extend([term, term])

    if category and category.strip() and category.strip().lower() != "all":
        query += " AND LOWER(category) = ?"
        params.append(category.strip().lower())

    query += " ORDER BY created_at DESC"

    with get_connection(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def update_document_chunk_count(
    document_id: str,
    chunk_count: int,
    db_path: Path | None = None,
) -> dict[str, Any] | None:
    """Update chunk count and timestamp after indexing or re-indexing."""
    now = utc_now_iso()
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE documents SET chunk_count = ?, updated_at = ? WHERE id = ?",
            (chunk_count, now, document_id),
        )
    return get_document_by_id(document_id, db_path=db_path)


def delete_document_record(document_id: str, db_path: Path | None = None) -> bool:
    """Delete a document record (and cascaded chunks). Returns True if a row was deleted."""
    with get_connection(db_path) as conn:
        cursor = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return cursor.rowcount > 0


def list_categories(db_path: Path | None = None) -> list[str]:
    """Return sorted unique document categories currently stored."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT category FROM documents ORDER BY LOWER(category) ASC"
        ).fetchall()
    return [row["category"] for row in rows]


# ---------------------------------------------------------------------------
# Chunk Operations
# ---------------------------------------------------------------------------

def insert_chunks(
    chunks: list[dict[str, Any]],
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Insert chunks and populate each dictionary with its SQLite-assigned faiss_id."""
    if not chunks:
        return []

    stored_chunks: list[dict[str, Any]] = []
    with get_connection(db_path) as conn:
        for chunk in chunks:
            cursor = conn.execute(
                """
                INSERT INTO chunks (
                    chunk_id, document_id, document_name, category,
                    page_number, chunk_index, text
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk["chunk_id"],
                    chunk["document_id"],
                    chunk["document_name"],
                    chunk["category"],
                    chunk.get("page_number"),
                    chunk["chunk_index"],
                    chunk["text"],
                ),
            )
            item = dict(chunk)
            item["faiss_id"] = int(cursor.lastrowid)
            stored_chunks.append(item)
    return stored_chunks


def get_chunks_by_document_id(
    document_id: str,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Fetch all chunks belonging to a single document ordered by chunk_index."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC",
            (document_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_chunks_by_faiss_ids(
    faiss_ids: list[int],
    db_path: Path | None = None,
) -> dict[int, dict[str, Any]]:
    """Look up chunk metadata by FAISS integer IDs."""
    if not faiss_ids:
        return {}

    placeholders = ",".join("?" for _ in faiss_ids)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            f"SELECT * FROM chunks WHERE faiss_id IN ({placeholders})",
            [int(fid) for fid in faiss_ids],
        ).fetchall()
    return {int(row["faiss_id"]): dict(row) for row in rows}


def get_all_chunks(db_path: Path | None = None) -> list[dict[str, Any]]:
    """Return every indexed chunk in the database."""
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM chunks ORDER BY faiss_id ASC").fetchall()
    return [dict(row) for row in rows]


def delete_chunks_by_document_id(
    document_id: str,
    db_path: Path | None = None,
) -> list[int]:
    """Delete chunks for a document and return their FAISS IDs for index removal."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT faiss_id FROM chunks WHERE document_id = ?",
            (document_id,),
        ).fetchall()
        faiss_ids = [int(row["faiss_id"]) for row in rows]
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
    return faiss_ids


# ---------------------------------------------------------------------------
# Settings Overrides Operations
# ---------------------------------------------------------------------------

def get_all_setting_overrides(db_path: Path | None = None) -> dict[str, str]:
    """Return persisted non-secret runtime settings overrides."""
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT key, value FROM app_settings").fetchall()
    return {row["key"]: row["value"] for row in rows}


def upsert_setting_overrides(
    updates: dict[str, Any],
    db_path: Path | None = None,
) -> None:
    """Persist runtime setting overrides in SQLite."""
    now = utc_now_iso()
    with get_connection(db_path) as conn:
        for key, value in updates.items():
            conn.execute(
                """
                INSERT INTO app_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, str(value), now),
            )


# ---------------------------------------------------------------------------
# Conversation, Message, Trace, and Benchmark Operations
# ---------------------------------------------------------------------------

def ensure_conversation(
    conversation_id: str,
    title: str,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Fetch an existing conversation or create it with the given title."""
    now = utc_now_iso()
    with get_connection(db_path) as conn:
        existing = conn.execute(
            "SELECT * FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if existing:
            return dict(existing)
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conversation_id, title[:80], now, now),
        )
        created = conn.execute(
            "SELECT * FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
    assert created is not None
    return dict(created)


def insert_message(
    message_id: str,
    conversation_id: str,
    role: str,
    content: str,
    query_id: str | None = None,
    mode: str = "crag",
    sources: list[dict[str, Any]] | None = None,
    db_path: Path | None = None,
) -> None:
    """Save a user or assistant message and update conversation timestamp."""
    now = utc_now_iso()
    sources_json = json.dumps(sources or [])
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO messages (
                id, conversation_id, role, content, query_id, mode, sources_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (message_id, conversation_id, role, content, query_id, mode, sources_json, now),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )


def list_conversations(db_path: Path | None = None) -> list[dict[str, Any]]:
    """List conversations ordered by most recently updated."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT c.*, COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_conversation_messages(
    conversation_id: str,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Retrieve all messages in a conversation in chronological order."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["sources"] = json.loads(item.pop("sources_json", "[]") or "[]")
        result.append(item)
    return result


def delete_conversation(conversation_id: str, db_path: Path | None = None) -> bool:
    """Delete a conversation and its messages."""
    with get_connection(db_path) as conn:
        cursor = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        return cursor.rowcount > 0


def save_pipeline_trace(trace_summary: dict[str, Any], db_path: Path | None = None) -> None:
    """Persist a pipeline execution trace for inspection and analytics."""
    now = trace_summary.get("created_at") or utc_now_iso()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO pipeline_traces (
                query_id, conversation_id, original_query, rewritten_query,
                mode, intent, internal_outcome, web_outcome, grounding_status,
                correction_attempted, web_fallback_used, avg_relevance_score,
                total_latency_ms, trace_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace_summary["query_id"],
                trace_summary.get("conversation_id"),
                trace_summary["original_query"],
                trace_summary.get("rewritten_query"),
                trace_summary.get("mode", "crag"),
                trace_summary.get("intent"),
                trace_summary.get("internal_outcome"),
                trace_summary.get("web_outcome"),
                trace_summary.get("grounding_status"),
                1 if trace_summary.get("correction_attempted") else 0,
                1 if trace_summary.get("web_fallback_used") else 0,
                float(trace_summary.get("avg_relevance_score", 0.0)),
                float(trace_summary.get("total_latency_ms", 0.0)),
                json.dumps(trace_summary["trace_data"]),
                now,
            ),
        )


def get_pipeline_trace(query_id: str, db_path: Path | None = None) -> dict[str, Any] | None:
    """Retrieve a full pipeline trace by query_id."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM pipeline_traces WHERE query_id = ?",
            (query_id,),
        ).fetchone()
    if not row:
        return None
    return json.loads(row["trace_json"])


def list_recent_traces(limit: int = 50, db_path: Path | None = None) -> list[dict[str, Any]]:
    """Return summary metadata of recent pipeline traces for analytics and UI selection."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT query_id, conversation_id, original_query, rewritten_query,
                   mode, intent, internal_outcome, web_outcome, grounding_status,
                   correction_attempted, web_fallback_used, avg_relevance_score,
                   total_latency_ms, created_at
            FROM pipeline_traces
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def save_benchmark_run(
    run_id: str,
    dataset_name: str,
    summary: dict[str, Any],
    details: dict[str, Any],
    db_path: Path | None = None,
) -> None:
    """Persist benchmark comparison results."""
    now = utc_now_iso()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO benchmark_runs (id, run_at, dataset_name, summary_json, details_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, now, dataset_name, json.dumps(summary), json.dumps(details)),
        )


def get_latest_benchmark_run(db_path: Path | None = None) -> dict[str, Any] | None:
    """Return the most recent benchmark comparison run if available."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM benchmark_runs ORDER BY run_at DESC LIMIT 1"
        ).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "run_at": row["run_at"],
        "dataset_name": row["dataset_name"],
        "summary": json.loads(row["summary_json"]),
        "details": json.loads(row["details_json"]),
    }

