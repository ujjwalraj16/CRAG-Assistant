"""FastAPI application entrypoint and router registration for CRAG AI (NEXUS AI)."""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.analytics import router as analytics_router
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.pipeline import router as pipeline_router
from app.api.settings import router as settings_router
from app.config import ensure_directories, get_public_settings, settings
from app.database.db import init_db
from app.database.models import get_all_setting_overrides, list_documents
from app.ingestion.indexer import FaissVectorStore
from app.ollama_client import check_ollama_health


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize required local directories and SQLite schema on startup."""
    ensure_directories()
    init_db()
    yield


app = FastAPI(
    title="CRAG AI - Adaptive Corrective RAG API",
    description="Modular Corrective Retrieval-Augmented Generation backend powered by FastAPI, FAISS, and Ollama.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(pipeline_router)
app.include_router(analytics_router)
app.include_router(settings_router)


@app.get("/api/health", tags=["Health"])
def health_check() -> dict[str, Any]:
    """Return service, database, FAISS index, Ollama, and Tavily configuration health."""
    ensure_directories()
    init_db()
    runtime_cfg = get_public_settings(get_all_setting_overrides())
    ollama_status = check_ollama_health(
        base_url=runtime_cfg["ollama_base_url"],
        model=runtime_cfg["ollama_model"],
    )
    docs = list_documents()
    vector_count = FaissVectorStore().total_vectors

    return {
        "status": "ok",
        "app_name": settings.app_name,
        "database": "connected",
        "document_count": len(docs),
        "indexed_vectors": vector_count,
        "tavily_configured": runtime_cfg["tavily_configured"],
        "ollama": ollama_status,
    }

