"""Application configuration and runtime settings management."""

from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    """Environment-backed application configuration."""

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "CRAG AI"
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Ollama Local LLM Configuration
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout_seconds: float = 90.0

    # Embedding Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Tavily Web Search (secret stays in backend environment only)
    tavily_api_key: str = ""
    web_search_enabled: bool = True
    web_search_max_results: int = 5

    # Retrieval and CRAG Defaults
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    max_correction_attempts: int = Field(default=1, ge=0, le=5)
    chunk_size: int = Field(default=500, ge=100, le=4000)
    chunk_overlap: int = Field(default=100, ge=0, le=1000)

    # Storage Paths
    data_dir: str = "data"
    upload_dir: str = "data/uploads"
    faiss_index_dir: str = "data/faiss_index"
    database_path: str = "data/crag.db"

    def resolve_path(self, configured_path: str) -> Path:
        """Resolve a path relative to the backend directory if not already absolute."""
        path = Path(configured_path)
        if path.is_absolute():
            return path
        return BACKEND_DIR / path

    @property
    def resolved_data_dir(self) -> Path:
        return self.resolve_path(self.data_dir)

    @property
    def resolved_upload_dir(self) -> Path:
        return self.resolve_path(self.upload_dir)

    @property
    def resolved_faiss_index_dir(self) -> Path:
        return self.resolve_path(self.faiss_index_dir)

    @property
    def resolved_database_path(self) -> Path:
        return self.resolve_path(self.database_path)

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def tavily_configured(self) -> bool:
        return bool(self.tavily_api_key.strip())


settings = Settings()


def ensure_directories(active_settings: Settings = settings) -> None:
    """Create local data directories required by SQLite, uploads, and FAISS."""
    active_settings.resolved_data_dir.mkdir(parents=True, exist_ok=True)
    active_settings.resolved_upload_dir.mkdir(parents=True, exist_ok=True)
    active_settings.resolved_faiss_index_dir.mkdir(parents=True, exist_ok=True)
    active_settings.resolved_database_path.parent.mkdir(parents=True, exist_ok=True)


def get_public_settings(overrides: dict[str, str] | None = None) -> dict[str, Any]:
    """Return non-secret runtime settings merged with persisted DB overrides."""
    merged: dict[str, Any] = {
        "ollama_model": settings.ollama_model,
        "ollama_base_url": settings.ollama_base_url,
        "embedding_model": settings.embedding_model,
        "tavily_configured": settings.tavily_configured,
        "web_search_enabled": settings.web_search_enabled,
        "retrieval_top_k": settings.retrieval_top_k,
        "similarity_threshold": settings.similarity_threshold,
        "max_correction_attempts": settings.max_correction_attempts,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    if not overrides:
        return merged

    int_keys = {"retrieval_top_k", "max_correction_attempts", "chunk_size", "chunk_overlap"}
    float_keys = {"similarity_threshold"}
    bool_keys = {"web_search_enabled"}
    str_keys = {"ollama_model", "ollama_base_url", "embedding_model"}

    for key, raw_val in overrides.items():
        if key not in merged:
            continue
        if key in int_keys:
            merged[key] = int(raw_val)
        elif key in float_keys:
            merged[key] = float(raw_val)
        elif key in bool_keys:
            merged[key] = raw_val.lower() in {"1", "true", "yes"}
        elif key in str_keys:
            merged[key] = str(raw_val)
    return merged

