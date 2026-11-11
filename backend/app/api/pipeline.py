"""Pipeline execution trace endpoints powering the CRAG Pipeline Monitor."""

from typing import Any

from fastapi import APIRouter, HTTPException, Query, status

from app.crag.trace_manager import get_recent_trace_summaries, load_trace
from app.schemas.pipeline import PipelineTrace


router = APIRouter(prefix="/api/pipeline", tags=["Pipeline"])


@router.get("", response_model=list[dict[str, Any]])
def list_pipeline_traces(limit: int = Query(default=25, ge=1, le=100)) -> list[dict[str, Any]]:
    """List recent pipeline execution summaries."""
    return get_recent_trace_summaries(limit=limit)


@router.get("/{query_id}", response_model=PipelineTrace)
def get_pipeline_trace_detail(query_id: str) -> PipelineTrace:
    """Fetch the complete 8-stage CRAG execution trace for a given query_id."""
    trace = load_trace(query_id)
    if trace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline trace for query '{query_id}' not found.",
        )
    return trace

