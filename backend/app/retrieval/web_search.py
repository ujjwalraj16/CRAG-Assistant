"""External web search fallback using the Tavily Search API with LLM relevance grading."""

from typing import Callable
import uuid

import httpx

from app.config import settings
from app.ollama_client import LLMCallable
from app.retrieval.relevance_grader import grade_text_relevance
from app.schemas.pipeline import (
    GradedChunk,
    WebSearchOutcome,
    WebSearchRawResult,
)


TAVILY_SEARCH_URL = "https://api.tavily.com/search"

WebSearchFn = Callable[[str, int, str], list[WebSearchRawResult]]


class TavilySearchError(RuntimeError):
    """Raised when the Tavily API request fails or returns an unexpected response."""


def fetch_tavily_results(
    query: str,
    max_results: int,
    api_key: str,
    timeout_seconds: float = 15.0,
) -> list[WebSearchRawResult]:
    """Query the Tavily Search REST API and normalize title, URL, and snippet content."""
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.post(TAVILY_SEARCH_URL, json=payload)
            response.raise_for_status()
            data = response.json()
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as exc:
        raise TavilySearchError(f"Tavily search request failed: {exc}") from exc

    raw_items = data.get("results") or []
    results: list[WebSearchRawResult] = []
    for item in raw_items:
        content = str(item.get("content") or "").strip()
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or url or "Web Result").strip()
        if not content or not url:
            continue
        results.append(
            WebSearchRawResult(
                title=title,
                url=url,
                content=content,
                score=float(item.get("score") or 0.0),
            )
        )
    return results


def run_web_fallback(
    query: str,
    web_search_enabled: bool | None = None,
    tavily_api_key: str | None = None,
    max_results: int | None = None,
    ollama_model: str | None = None,
    ollama_base_url: str | None = None,
    search_fn: WebSearchFn | None = None,
    grade_llm_fn: LLMCallable | None = None,
) -> WebSearchOutcome:
    """
    Execute external web search when internal knowledge is insufficient, then grade results.

    Never claims a search ran if disabled, unconfigured, or unreachable.
    """
    enabled = settings.web_search_enabled if web_search_enabled is None else web_search_enabled
    if not enabled:
        return WebSearchOutcome(
            status="SKIPPED",
            executed=False,
            query_used=None,
            explanation="External web search is disabled in configuration.",
        )

    api_key = (settings.tavily_api_key if tavily_api_key is None else tavily_api_key).strip()
    if not api_key and search_fn is None:
        return WebSearchOutcome(
            status="UNAVAILABLE",
            executed=False,
            query_used=None,
            explanation="Tavily API key is not configured; continuing with internal documents only.",
        )

    cleaned_query = query.strip()
    if not cleaned_query:
        return WebSearchOutcome(
            status="SKIPPED",
            executed=False,
            query_used=None,
            explanation="Cannot run web search for an empty query.",
        )

    limit = max_results or settings.web_search_max_results
    try:
        raw_results = (
            search_fn(cleaned_query, limit, api_key)
            if search_fn is not None
            else fetch_tavily_results(cleaned_query, max_results=limit, api_key=api_key)
        )
    except Exception as exc:
        return WebSearchOutcome(
            status="ERROR",
            executed=True,
            query_used=cleaned_query,
            explanation="External web search failed; continuing with available internal evidence.",
            error=str(exc),
        )

    if not raw_results:
        return WebSearchOutcome(
            status="NO_RESULTS",
            executed=True,
            query_used=cleaned_query,
            raw_results_count=0,
            explanation="Tavily web search returned no usable snippets for the query.",
        )

    graded_results: list[GradedChunk] = []
    ollama_available = True
    batch_id = uuid.uuid4().hex[:8]

    for idx, item in enumerate(raw_results):
        grade, parse_fallback, reachable = grade_text_relevance(
            query=cleaned_query,
            evidence_text=item.content,
            source_name=f"{item.title} ({item.url})",
            model=ollama_model,
            base_url=ollama_base_url,
            llm_fn=grade_llm_fn,
        )
        if not reachable:
            ollama_available = False

        graded_results.append(
            GradedChunk(
                chunk_id=f"web_{batch_id}_{idx}",
                document_id=f"web:{item.url}",
                document_name=item.title,
                category="External Web",
                page_number=None,
                chunk_index=idx,
                text=item.content,
                similarity_score=round(float(item.score), 4),
                source_type="web",
                url=item.url,
                title=item.title,
                relevance=grade.relevance,
                confidence=grade.confidence,
                reason=grade.reason,
                parse_fallback=parse_fallback,
            )
        )

    relevant_count = sum(1 for item in graded_results if item.relevance == "relevant")
    explanation = (
        f"Web search retrieved {len(raw_results)} results; "
        f"{relevant_count} graded as relevant."
    )
    if not ollama_available:
        explanation = f"Ollama unavailable while grading web results; {explanation}"

    return WebSearchOutcome(
        status="SUCCESS",
        executed=True,
        query_used=cleaned_query,
        raw_results_count=len(raw_results),
        graded_results=graded_results,
        relevant_count=relevant_count,
        explanation=explanation,
        ollama_available=ollama_available,
    )

