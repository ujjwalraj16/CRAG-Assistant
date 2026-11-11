"""Analytics endpoints reporting real metrics computed from stored traces and benchmark runs."""

from collections import defaultdict
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.config import get_public_settings
from app.database.db import get_connection
from app.database.models import get_all_setting_overrides, get_latest_benchmark_run
from app.evaluation.benchmark import run_benchmark
from app.ollama_client import check_ollama_health


router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


@router.get("/overview")
def get_analytics_overview() -> dict[str, Any]:
    """Compute real query, retrieval, correction, web fallback, and latency metrics."""
    with get_connection() as conn:
        doc_stats = conn.execute(
            "SELECT COUNT(*) AS doc_count, COALESCE(SUM(chunk_count), 0) AS chunk_count FROM documents"
        ).fetchone()
        traces = conn.execute(
            """
            SELECT query_id, original_query, mode, internal_outcome, web_outcome,
                   grounding_status, correction_attempted, web_fallback_used,
                   avg_relevance_score, total_latency_ms, created_at
            FROM pipeline_traces
            ORDER BY created_at ASC
            """
        ).fetchall()

    trace_rows = [dict(row) for row in traces]
    total_queries = len(trace_rows)
    crag_rows = [row for row in trace_rows if row["mode"] == "crag"]
    basic_rows = [row for row in trace_rows if row["mode"] == "basic_rag"]

    if total_queries == 0:
        return {
            "has_data": False,
            "total_queries": 0,
            "crag_queries": 0,
            "basic_rag_queries": 0,
            "document_count": int(doc_stats["doc_count"]),
            "total_chunks": int(doc_stats["chunk_count"]),
            "avg_latency_ms": 0.0,
            "internal_success_rate": 0.0,
            "correction_frequency": 0.0,
            "web_fallback_frequency": 0.0,
            "avg_retrieval_relevance": 0.0,
            "grounding_breakdown": [],
            "correction_distribution": [],
            "queries_over_time": [],
            "latency_history": [],
        }

    avg_latency_ms = round(
        sum(float(row["total_latency_ms"]) for row in trace_rows) / total_queries, 2
    )

    crag_count = len(crag_rows)
    if crag_count > 0:
        internal_correct = sum(1 for row in crag_rows if row["internal_outcome"] == "CORRECT")
        corrections = sum(1 for row in crag_rows if int(row["correction_attempted"]) == 1)
        web_fallbacks = sum(1 for row in crag_rows if int(row["web_fallback_used"]) == 1)
        internal_success_rate = round((internal_correct / crag_count) * 100.0, 1)
        correction_frequency = round((corrections / crag_count) * 100.0, 1)
        web_fallback_frequency = round((web_fallbacks / crag_count) * 100.0, 1)
        avg_relevance = round(
            sum(float(row["avg_relevance_score"]) for row in crag_rows) / crag_count, 4
        )
    else:
        internal_success_rate = 0.0
        correction_frequency = 0.0
        web_fallback_frequency = 0.0
        avg_relevance = 0.0

    grounding_counts: dict[str, int] = defaultdict(int)
    outcome_counts: dict[str, int] = defaultdict(int)
    daily_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in crag_rows:
        grounding_counts[row["grounding_status"] or "Skipped"] += 1
        outcome_counts[row["internal_outcome"] or "INCORRECT"] += 1

    for row in trace_rows:
        date_key = str(row["created_at"])[:10]
        daily_buckets[date_key].append(row)

    queries_over_time = [
        {
            "date": date_key,
            "queries": len(bucket),
            "avg_latency_ms": round(
                sum(float(item["total_latency_ms"]) for item in bucket) / len(bucket), 2
            ),
        }
        for date_key, bucket in sorted(daily_buckets.items())
    ]

    latency_history = [
        {
            "label": f"Q{idx}",
            "query_id": row["query_id"],
            "query": row["original_query"][:40],
            "mode": row["mode"],
            "latency_ms": round(float(row["total_latency_ms"]), 2),
        }
        for idx, row in enumerate(trace_rows[-20:], start=1)
    ]

    return {
        "has_data": True,
        "total_queries": total_queries,
        "crag_queries": crag_count,
        "basic_rag_queries": len(basic_rows),
        "document_count": int(doc_stats["doc_count"]),
        "total_chunks": int(doc_stats["chunk_count"]),
        "avg_latency_ms": avg_latency_ms,
        "internal_success_rate": internal_success_rate,
        "correction_frequency": correction_frequency,
        "web_fallback_frequency": web_fallback_frequency,
        "avg_retrieval_relevance": avg_relevance,
        "grounding_breakdown": [
            {"name": status_name, "value": count}
            for status_name, count in grounding_counts.items()
        ],
        "correction_distribution": [
            {"name": outcome, "value": outcome_counts.get(outcome, 0)}
            for outcome in ("CORRECT", "AMBIGUOUS", "INCORRECT")
        ],
        "queries_over_time": queries_over_time,
        "latency_history": latency_history,
    }


@router.get("/comparison")
def get_comparison_report() -> dict[str, Any]:
    """Return the latest persisted Basic RAG vs CRAG benchmark report if available."""
    latest = get_latest_benchmark_run()
    if latest is None:
        return {
            "available": False,
            "run": None,
            "message": "No benchmark evaluation run recorded yet. Run the evaluation benchmark to generate real comparison metrics.",
        }
    return {
        "available": True,
        "run": latest,
        "message": f"Loaded benchmark run '{latest['id']}' from {latest['run_at']}.",
    }


class BenchmarkTriggerRequest(BaseModel):
    """Optional parameters for triggering a live Basic RAG vs CRAG benchmark run."""

    top_k: int = Field(default=3, ge=1, le=10)
    web_search_enabled: bool = False
    use_llm_judge: bool = True


@router.post("/benchmark")
def trigger_benchmark_run(payload: BenchmarkTriggerRequest | None = None) -> dict[str, Any]:
    """Execute the labelled benchmark dataset with local Ollama and persist real results."""
    req = payload or BenchmarkTriggerRequest()
    runtime_cfg = get_public_settings(get_all_setting_overrides())
    health = check_ollama_health(
        base_url=runtime_cfg["ollama_base_url"],
        model=runtime_cfg["ollama_model"],
    )
    if not health["reachable"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Ollama is not reachable at {runtime_cfg['ollama_base_url']}. "
                "Start Ollama (`ollama serve`) before running a live benchmark."
            ),
        )

    report = run_benchmark(
        top_k=req.top_k,
        web_search_enabled=req.web_search_enabled,
        use_llm_judge=req.use_llm_judge,
    )
    latest = get_latest_benchmark_run()
    return {
        "available": True,
        "run": latest,
        "message": f"Completed live benchmark '{report['run_id']}'.",
    }

