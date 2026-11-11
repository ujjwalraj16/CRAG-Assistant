"""Settings endpoints for viewing and updating non-secret runtime configuration."""

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from app.config import get_public_settings
from app.database.models import get_all_setting_overrides, upsert_setting_overrides


router = APIRouter(prefix="/api/settings", tags=["Settings"])


class SettingsUpdateRequest(BaseModel):
    """Editable non-secret pipeline and model configuration parameters."""

    ollama_model: str = Field(..., min_length=1, max_length=120)
    ollama_base_url: str = Field(..., min_length=1, max_length=250)
    embedding_model: str = Field(..., min_length=1, max_length=200)
    web_search_enabled: bool = True
    retrieval_top_k: int = Field(..., ge=1, le=20)
    similarity_threshold: float = Field(..., ge=0.0, le=1.0)
    max_correction_attempts: int = Field(..., ge=0, le=5)
    chunk_size: int = Field(..., ge=100, le=4000)
    chunk_overlap: int = Field(..., ge=0, le=1000)

    @model_validator(mode="after")
    def validate_chunk_bounds(self) -> "SettingsUpdateRequest":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size.")
        if not self.ollama_base_url.startswith(("http://", "https://")):
            raise ValueError("ollama_base_url must start with http:// or https://.")
        return self


@router.get("")
def read_settings() -> dict[str, Any]:
    """Return active non-secret runtime settings and Tavily API key configuration status."""
    overrides = get_all_setting_overrides()
    return get_public_settings(overrides)


@router.put("")
def update_settings(payload: SettingsUpdateRequest) -> dict[str, Any]:
    """Persist updated non-secret configuration values to SQLite."""
    updates = {
        "ollama_model": payload.ollama_model.strip(),
        "ollama_base_url": payload.ollama_base_url.strip().rstrip("/"),
        "embedding_model": payload.embedding_model.strip(),
        "web_search_enabled": str(payload.web_search_enabled).lower(),
        "retrieval_top_k": str(payload.retrieval_top_k),
        "similarity_threshold": str(payload.similarity_threshold),
        "max_correction_attempts": str(payload.max_correction_attempts),
        "chunk_size": str(payload.chunk_size),
        "chunk_overlap": str(payload.chunk_overlap),
    }
    try:
        upsert_setting_overrides(updates)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to persist settings: {exc}",
        ) from exc

    return get_public_settings(get_all_setting_overrides())

