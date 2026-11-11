"""Schemas for chat requests, responses, and conversation history."""

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.pipeline import (
    PipelineMode,
    QueryIntent,
    RetrievalOutcome,
    SourceCitation,
    TraceGroundingStatus,
    WebSearchStatus,
)


class ChatRequest(BaseModel):
    """Incoming user message payload for POST /api/chat."""

    query: str = Field(..., min_length=1, max_length=4000)
    conversation_id: str | None = None
    mode: PipelineMode = "crag"
    category: str | None = None
    document_ids: list[str] | None = None
    use_web_search: bool | None = None


class ChatResponse(BaseModel):
    """Structured chat response including citations and pipeline status summary."""

    conversation_id: str
    query_id: str
    mode: PipelineMode
    intent: QueryIntent
    answer: str
    abstained: bool = False
    citations: list[SourceCitation] = Field(default_factory=list)
    internal_outcome: RetrievalOutcome | None = None
    web_status: WebSearchStatus | None = None
    grounding_status: TraceGroundingStatus = "Skipped"
    rewritten_query: str | None = None
    total_latency_ms: float = 0.0


class ChatMessageRecord(BaseModel):
    """Persisted message within a conversation."""

    id: str
    conversation_id: str
    role: Literal["user", "assistant"]
    content: str
    query_id: str | None = None
    mode: str = "crag"
    sources: list[SourceCitation] = Field(default_factory=list)
    created_at: str


class ConversationSummary(BaseModel):
    """Sidebar conversation item metadata."""

    id: str
    title: str
    message_count: int = 0
    created_at: str
    updated_at: str


class ConversationDetailResponse(BaseModel):
    """Full message history for a single conversation."""

    conversation_id: str
    messages: list[ChatMessageRecord] = Field(default_factory=list)

