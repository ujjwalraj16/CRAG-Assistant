"""Pipeline execution trace builder and SQLite persistence helpers."""

from pathlib import Path
from typing import Any

from app.database.models import get_pipeline_trace, list_recent_traces, save_pipeline_trace, utc_now_iso
from app.schemas.pipeline import PipelineStageTrace, PipelineTrace


class TraceBuilder:
    """Collects real-time stage timings and metadata during a single query execution."""

    def __init__(self, query_id: str, original_query: str, mode: str, conversation_id: str | None = None) -> None:
        self.query_id = query_id
        self.original_query = original_query
        self.mode = mode
        self.conversation_id = conversation_id
        self.stages: list[PipelineStageTrace] = []
        self.stage_timings_ms: dict[str, float] = {}

    def record_stage(
        self,
        stage_number: int,
        name: str,
        status: str,
        duration_ms: float,
        explanation: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Append an executed or skipped stage record with observed timings."""
        rounded_ms = round(float(duration_ms), 2)
        self.stage_timings_ms[name] = rounded_ms
        self.stages.append(
            PipelineStageTrace(
                stage_number=stage_number,
                name=name,
                status=status,
                duration_ms=rounded_ms,
                explanation=explanation,
                details=details or {},
            )
        )


def persist_trace(trace: PipelineTrace, db_path: Path | None = None) -> None:
    """Save a completed PipelineTrace to SQLite for Pipeline Monitor and Analytics."""
    avg_relevance = 0.0
    if trace.relevance_grades:
        relevant_scores = [
            item.confidence for item in trace.relevance_grades if item.relevance == "relevant"
        ]
        if relevant_scores:
            avg_relevance = round(sum(relevant_scores) / len(relevant_scores), 4)
        else:
            avg_relevance = round(
                sum(item.confidence for item in trace.relevance_grades) / len(trace.relevance_grades),
                4,
            )

    correction_attempted = any(
        stage.name == "Query Correction" and stage.status == "Completed"
        for stage in trace.stages
    )
    web_fallback_used = bool(
        trace.web_outcome is not None
        and trace.web_outcome.executed
        and trace.web_outcome.status in {"SUCCESS", "NO_RESULTS"}
    )

    save_pipeline_trace(
        {
            "query_id": trace.query_id,
            "conversation_id": trace.conversation_id,
            "original_query": trace.original_query,
            "rewritten_query": trace.rewritten_query,
            "mode": trace.mode,
            "intent": trace.intent,
            "internal_outcome": trace.internal_outcome,
            "web_outcome": trace.web_outcome.status if trace.web_outcome else "SKIPPED",
            "grounding_status": trace.grounding_status,
            "correction_attempted": correction_attempted,
            "web_fallback_used": web_fallback_used,
            "avg_relevance_score": avg_relevance,
            "total_latency_ms": trace.total_latency_ms,
            "trace_data": trace.model_dump(),
            "created_at": trace.created_at or utc_now_iso(),
        },
        db_path=db_path,
    )


def load_trace(query_id: str, db_path: Path | None = None) -> PipelineTrace | None:
    """Load a persisted PipelineTrace from SQLite by query_id."""
    raw = get_pipeline_trace(query_id, db_path=db_path)
    if raw is None:
        return None
    return PipelineTrace.model_validate(raw)


def get_recent_trace_summaries(limit: int = 50, db_path: Path | None = None) -> list[dict[str, Any]]:
    """List recent trace summaries for UI trace selection and analytics."""
    return list_recent_traces(limit=limit, db_path=db_path)

