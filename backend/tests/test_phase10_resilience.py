"""Phase 10 resilience tests: end-to-end failure scenarios and graceful degradation."""

from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest

from app.config import settings
from app.crag.pipeline import run_crag_pipeline
from app.database.db import init_db
from app.main import app
from app.ollama_client import OllamaUnavailableError


def _zero_embedder(texts: list[str], model_name: str | None = None) -> np.ndarray:
    vec = np.zeros((len(texts), 8), dtype=np.float32)
    vec[:, 0] = 1.0
    return vec


@pytest.fixture()
def isolated_api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "faiss_index_dir", str(tmp_path / "faiss"))
    monkeypatch.setattr(settings, "database_path", str(tmp_path / "crag.db"))
    monkeypatch.setattr(settings, "tavily_api_key", "")
    init_db(tmp_path / "crag.db")

    monkeypatch.setattr("app.ingestion.indexer.embed_texts", _zero_embedder)
    monkeypatch.setattr("app.retrieval.retriever.embed_texts", _zero_embedder)

    with TestClient(app) as client:
        yield client


def test_crag_pipeline_empty_index_and_missing_tavily_abstains_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "empty.db"
    index_dir = tmp_path / "empty_faiss"
    init_db(db_path)
    monkeypatch.setattr(settings, "tavily_api_key", "")

    trace = run_crag_pipeline(
        query="What is the warranty period on AeroBook laptops?",
        web_search_enabled=True,
        db_path=db_path,
        index_dir=index_dir,
        embed_fn=_zero_embedder,
    )
    assert trace.internal_outcome == "INCORRECT"
    assert trace.web_outcome is not None
    assert trace.web_outcome.status == "UNAVAILABLE"
    assert trace.web_outcome.executed is False
    assert trace.abstained is True
    assert trace.grounding_status == "Abstained"
    assert len(trace.stages) == 8


def test_api_upload_empty_file_and_chat_when_ollama_down(
    isolated_api_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    # Empty file upload returns clean 400 Bad Request.
    empty_res = isolated_api_client.post(
        "/api/documents/upload",
        data={"category": "General"},
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert empty_res.status_code == 400

    # Upload a valid doc so retrieval succeeds, then simulate Ollama down.
    ok_res = isolated_api_client.post(
        "/api/documents/upload",
        data={"category": "Laptops"},
        files={"file": ("laptop.txt", b"AeroBook has a 2-year battery warranty.", "text/plain")},
    )
    assert ok_res.status_code == 201

    def down_ollama(*args, **kwargs):
        raise OllamaUnavailableError("Connection refused")

    monkeypatch.setattr("app.retrieval.relevance_grader.call_ollama", down_ollama)
    monkeypatch.setattr("app.crag.generator.call_ollama", down_ollama)

    # Chat endpoint must degrade gracefully instead of crashing with HTTP 500.
    chat_res = isolated_api_client.post(
        "/api/chat",
        json={"query": "What is the AeroBook battery warranty?", "mode": "crag"},
    )
    assert chat_res.status_code == 200
    body = chat_res.json()
    assert body["abstained"] is True
    assert "ollama" in body["answer"].lower()

    trace = isolated_api_client.get(f"/api/pipeline/{body['query_id']}").json()
    assert any(stage["status"] == "Failed" for stage in trace["stages"])

