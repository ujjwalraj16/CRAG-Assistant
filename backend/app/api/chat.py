"""Chat endpoints executing CRAG or Basic RAG and persisting conversation history."""

import uuid

from fastapi import APIRouter, HTTPException, status

from app.crag.pipeline import run_basic_rag_pipeline, run_crag_pipeline
from app.database.models import (
    delete_conversation,
    ensure_conversation,
    get_conversation_messages,
    insert_message,
    list_conversations,
)
from app.schemas.chat import (
    ChatMessageRecord,
    ChatRequest,
    ChatResponse,
    ConversationDetailResponse,
    ConversationSummary,
)


router = APIRouter(prefix="/api/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
def send_chat_message(payload: ChatRequest) -> ChatResponse:
    """Process a user query via CRAG or Basic RAG and persist messages and execution trace."""
    trimmed_query = payload.query.strip()
    if not trimmed_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must not be empty or whitespace.",
        )

    conversation_id = payload.conversation_id or str(uuid.uuid4())
    title = trimmed_query if len(trimmed_query) <= 60 else f"{trimmed_query[:57]}..."
    ensure_conversation(conversation_id=conversation_id, title=title)

    insert_message(
        message_id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        role="user",
        content=trimmed_query,
        mode=payload.mode,
    )

    if payload.mode == "basic_rag":
        trace = run_basic_rag_pipeline(
            query=trimmed_query,
            conversation_id=conversation_id,
            category=payload.category,
            document_ids=payload.document_ids,
        )
    else:
        trace = run_crag_pipeline(
            query=trimmed_query,
            conversation_id=conversation_id,
            category=payload.category,
            document_ids=payload.document_ids,
            web_search_enabled=payload.use_web_search,
        )

    insert_message(
        message_id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        role="assistant",
        content=trace.answer,
        query_id=trace.query_id,
        mode=payload.mode,
        sources=[citation.model_dump() for citation in trace.citations],
    )

    return ChatResponse(
        conversation_id=conversation_id,
        query_id=trace.query_id,
        mode=trace.mode,
        intent=trace.intent,
        answer=trace.answer,
        abstained=trace.abstained,
        citations=trace.citations,
        internal_outcome=trace.internal_outcome,
        web_status=trace.web_outcome.status if trace.web_outcome else None,
        grounding_status=trace.grounding_status,
        rewritten_query=trace.rewritten_query,
        total_latency_ms=trace.total_latency_ms,
    )


@router.get("/conversations", response_model=list[ConversationSummary])
def get_conversations() -> list[ConversationSummary]:
    """List saved chat conversations ordered by recent activity."""
    rows = list_conversations()
    return [ConversationSummary(**row) for row in rows]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
def get_conversation(conversation_id: str) -> ConversationDetailResponse:
    """Return all messages in a conversation."""
    rows = get_conversation_messages(conversation_id)
    messages = [ChatMessageRecord(**row) for row in rows]
    return ConversationDetailResponse(
        conversation_id=conversation_id,
        messages=messages,
    )


@router.delete("/conversations/{conversation_id}")
def clear_conversation(conversation_id: str) -> dict[str, str]:
    """Delete a conversation and its stored messages."""
    deleted = delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )
    return {"conversation_id": conversation_id, "message": "Conversation cleared."}

