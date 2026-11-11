"""Phase 7 tests: FastAPI REST endpoints, document lifecycle, chat trace integration, analytics, and settings."""

import json
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest

from app.config import settings
from app.database.db import init_db
from app.main import app
from app.schemas.pipeline import PipelineTrace


TERMS = [
    "aerobook",
    "battery",
    "warranty",
    "years",
    "refund",
    "policy",
    "charger",
]


def fake_embed_texts(texts: list[str], model_name: str | None = None) -> np.ndarray:
    vectors = np.zeros((len(texts), len(TERMS) + 1), dtype=np.float32)
    for row_idx, text in enumerate(texts):
        lowered = text.lower()
        for col_idx, term in enumerate(TERMS):
            vectors[row_idx, col_idx] = float(lowered.count(term))
        vectors[row_idx, -1] = 0.05
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


@pytest.fixture()
def isolated_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Provide a FastAPI TestClient isolated to temporary SQLite, upload, and FAISS folders."""
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "faiss_index_dir", str(tmp_path / "faiss"))
    monkeypatch.setattr(settings, "database_path", str(tmp_path / "crag.db"))
    monkeypatch.setattr(settings, "tavily_api_key", "")
    init_db(tmp_path / "crag.db")

    monkeypatch.setattr("app.ingestion.indexer.embed_texts", fake_embed_texts)
    monkeypatch.setattr("app.retrieval.retriever.embed_texts", fake_embed_texts)

    with TestClient(app) as client:
        yield client


def test_health_and_empty_analytics(isolated_client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "app.main.check_ollama_health",
        lambda base_url=None, model=None: {
            "reachable": False,
            "base_url": "http://127.0.0.1:11434",
            "configured_model": "llama3.2:3b",
            "model_available": False,
            "available_models": [],
        },
    )

    health = isolated_client.get("/api/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert body["tavily_configured"] is False

    overview = isolated_client.get("/api/analytics/overview")
    assert overview.status_code == 200
    assert overview.json()["has_data"] is False
    assert overview.json()["total_queries"] == 0

    comparison = isolated_client.get("/api/analytics/comparison")
    assert comparison.status_code == 200
    assert comparison.json()["available"] is False


def test_document_upload_list_reindex_and_delete(isolated_client: TestClient):
    upload_res = isolated_client.post(
        "/api/documents/upload",
        data={"category": "Laptops"},
        files={"file": ("warranty.txt", b"AeroBook 14 includes a 2-year battery warranty.", "text/plain")},
    )
    assert upload_res.status_code == 201
    doc = upload_res.json()["document"]
    assert doc["filename"] == "warranty.txt"
    assert doc["category"] == "Laptops"
    assert doc["chunk_count"] >= 1
    doc_id = doc["id"]

    # Duplicate upload should be rejected with 409 Conflict.
    dup_res = isolated_client.post(
        "/api/documents/upload",
        data={"category": "Laptops"},
        files={"file": ("warranty.txt", b"AeroBook 14 includes a 2-year battery warranty.", "text/plain")},
    )
    assert dup_res.status_code == 409

    # Unsupported extension rejected with 400.
    bad_res = isolated_client.post(
        "/api/documents/upload",
        data={"category": "Laptops"},
        files={"file": ("malware.exe", b"binary", "application/octet-stream")},
    )
    assert bad_res.status_code == 400

    list_res = isolated_client.get("/api/documents", params={"category": "Laptops", "search": "warranty"})
    assert list_res.status_code == 200
    assert list_res.json()["total"] == 1
    assert "Laptops" in list_res.json()["categories"]

    reindex_res = isolated_client.post(f"/api/documents/{doc_id}/reindex")
    assert reindex_res.status_code == 200
    assert reindex_res.json()["document"]["id"] == doc_id

    del_res = isolated_client.delete(f"/api/documents/{doc_id}")
    assert del_res.status_code == 200
    assert del_res.json()["deleted_chunks"] >= 1

    assert isolated_client.get("/api/documents").json()["total"] == 0


def test_chat_and_pipeline_trace_and_analytics_flow(
    isolated_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    isolated_client.post(
        "/api/documents/upload",
        data={"category": "Laptops"},
        files={"file": ("aerobook.txt", b"AeroBook 14 comes with a 2-year battery warranty.", "text/plain")},
    )

    def mock_call_ollama(
        prompt: str,
        system_prompt: str | None = None,
        expect_json: bool = False,
        **kwargs,
    ) -> str:
        if system_prompt and "relevance evaluator" in system_prompt.lower():
            return json.dumps({"relevance": "relevant", "confidence": 0.93, "reason": "Direct match."})
        if system_prompt and "grounding verifier" in system_prompt.lower():
            return json.dumps({"status": "Supported", "explanation": "Grounded in Source 1.", "unsupported_claims": []})
        return "The AeroBook 14 comes with a 2-year battery warranty [Source 1]."

    monkeypatch.setattr("app.retrieval.relevance_grader.call_ollama", mock_call_ollama)
    monkeypatch.setattr("app.crag.generator.call_ollama", mock_call_ollama)
    monkeypatch.setattr("app.crag.grounding_checker.call_ollama", mock_call_ollama)

    chat_res = isolated_client.post(
        "/api/chat",
        json={"query": "What is the AeroBook battery warranty?", "mode": "crag"},
    )
    assert chat_res.status_code == 200
    chat_body = chat_res.json()
    assert chat_body["grounding_status"] == "Supported"
    assert chat_body["internal_outcome"] == "CORRECT"
    assert len(chat_body["citations"]) == 1
    query_id = chat_body["query_id"]
    conv_id = chat_body["conversation_id"]

    # Pipeline trace endpoint must expose the real persisted trace.
    trace_res = isolated_client.get(f"/api/pipeline/{query_id}")
    assert trace_res.status_code == 200
    parsed_trace = PipelineTrace.model_validate(trace_res.json())
    assert len(parsed_trace.stages) == 8
    assert parsed_trace.answer == chat_body["answer"]

    # Conversation history endpoints.
    convs = isolated_client.get("/api/chat/conversations").json()
    assert len(convs) == 1
    assert convs[0]["id"] == conv_id
    msgs = isolated_client.get(f"/api/chat/conversations/{conv_id}").json()["messages"]
    assert len(msgs) == 2

    # Analytics reflects real executed query.
    overview = isolated_client.get("/api/analytics/overview").json()
    assert overview["has_data"] is True
    assert overview["total_queries"] == 1
    assert overview["internal_success_rate"] == 100.0

    # Clear conversation.
    del_conv = isolated_client.delete(f"/api/chat/conversations/{conv_id}")
    assert del_conv.status_code == 200


def test_settings_read_update_and_validation(isolated_client: TestClient):
    initial = isolated_client.get("/api/settings")
    assert initial.status_code == 200
    data = initial.json()
    assert "tavily_api_key" not in data
    assert data["tavily_configured"] is False

    updated = isolated_client.put(
        "/api/settings",
        json={
            "ollama_model": "llama3.2:3b",
            "ollama_base_url": "http://127.0.0.1:11434",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "web_search_enabled": False,
            "retrieval_top_k": 7,
            "similarity_threshold": 0.3,
            "max_correction_attempts": 2,
            "chunk_size": 600,
            "chunk_overlap": 120,
        },
    )
    assert updated.status_code == 200
    assert updated.json()["retrieval_top_k"] == 7
    assert updated.json()["web_search_enabled"] is False

    # Invalid chunk_overlap >= chunk_size should fail validation with 422.
    invalid = isolated_client.put(
        "/api/settings",
        json={
            "ollama_model": "llama3.2:3b",
            "ollama_base_url": "http://127.0.0.1:11434",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "web_search_enabled": True,
            "retrieval_top_k": 5,
            "similarity_threshold": 0.25,
            "max_correction_attempts": 1,
            "chunk_size": 200,
            "chunk_overlap": 200,
        },
    )
    assert invalid.status_code == 422

