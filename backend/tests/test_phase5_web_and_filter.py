"""Phase 5 tests: Tavily web search fallback, graceful missing-key handling, and evidence filtering."""

import json

from app.retrieval.evidence_filter import filter_evidence
from app.retrieval.web_search import TavilySearchError, run_web_fallback
from app.schemas.pipeline import GradedChunk, WebSearchRawResult


def _make_graded(
    chunk_id: str,
    text: str,
    relevance: str = "relevant",
    confidence: float = 0.85,
    source_type: str = "internal",
    url: str | None = None,
) -> GradedChunk:
    return GradedChunk(
        chunk_id=chunk_id,
        document_id="doc-1" if source_type == "internal" else f"web:{url}",
        document_name="manual.pdf" if source_type == "internal" else "Web Page",
        category="Electronics" if source_type == "internal" else "External Web",
        page_number=2 if source_type == "internal" else None,
        chunk_index=0,
        text=text,
        similarity_score=0.75,
        source_type=source_type,
        url=url,
        title="Web Page" if source_type == "web" else None,
        relevance=relevance,
        confidence=confidence,
        reason="Test grade",
    )


def test_web_fallback_missing_api_key_never_fakes_search():
    outcome = run_web_fallback(
        query="AeroBook 14 HDMI port version",
        web_search_enabled=True,
        tavily_api_key="",
    )
    assert outcome.status == "UNAVAILABLE"
    assert outcome.executed is False
    assert outcome.graded_results == []
    assert "not configured" in outcome.explanation.lower()


def test_web_fallback_disabled_returns_skipped():
    outcome = run_web_fallback(
        query="AeroBook 14 HDMI port version",
        web_search_enabled=False,
        tavily_api_key="tvly-test",
    )
    assert outcome.status == "SKIPPED"
    assert outcome.executed is False


def test_web_fallback_grades_and_filters_external_results():
    def mock_search(query: str, limit: int, api_key: str) -> list[WebSearchRawResult]:
        return [
            WebSearchRawResult(
                title="AeroBook 14 Official Specs",
                url="https://example.com/aerobook-specs",
                content="AeroBook 14 includes an HDMI 2.1 port and two Thunderbolt 4 ports.",
                score=0.92,
            ),
            WebSearchRawResult(
                title="Unrelated Gaming Monitor Review",
                url="https://example.com/monitor",
                content="This 27-inch monitor supports DisplayPort 1.4.",
                score=0.55,
            ),
        ]

    def mock_grade(prompt: str, sys_prompt: str | None, expect_json: bool) -> str:
        if "AeroBook 14 includes an HDMI 2.1" in prompt:
            return json.dumps({"relevance": "relevant", "confidence": 0.91, "reason": "Direct HDMI spec."})
        return json.dumps({"relevance": "irrelevant", "confidence": 0.88, "reason": "About monitor, not AeroBook."})

    outcome = run_web_fallback(
        query="AeroBook 14 HDMI port version",
        web_search_enabled=True,
        tavily_api_key="tvly-test",
        search_fn=mock_search,
        grade_llm_fn=mock_grade,
    )
    assert outcome.status == "SUCCESS"
    assert outcome.executed is True
    assert outcome.raw_results_count == 2
    assert outcome.relevant_count == 1
    assert len(outcome.graded_results) == 2


def test_web_fallback_handles_tavily_errors_gracefully():
    def failing_search(query: str, limit: int, api_key: str) -> list[WebSearchRawResult]:
        raise TavilySearchError("503 Service Unavailable")

    outcome = run_web_fallback(
        query="AeroBook specs",
        web_search_enabled=True,
        tavily_api_key="tvly-test",
        search_fn=failing_search,
    )
    assert outcome.status == "ERROR"
    assert outcome.executed is True
    assert outcome.error is not None
    assert outcome.graded_results == []


def test_evidence_filter_removes_irrelevant_low_conf_and_duplicates():
    internal = [
        _make_graded("c1", "AeroBook battery warranty is 2 years.", relevance="relevant", confidence=0.92),
        _make_graded("c2", "AeroBook   battery warranty is 2 years.", relevance="relevant", confidence=0.70),
        _make_graded("c3", "Marginally related warranty note.", relevance="relevant", confidence=0.30),
        _make_graded("c4", "Camera has 12MP sensor.", relevance="irrelevant", confidence=0.85),
    ]
    web = [
        _make_graded(
            "w1",
            "Official support page confirms 24-month battery warranty.",
            relevance="relevant",
            confidence=0.88,
            source_type="web",
            url="https://support.example.com/warranty",
        ),
        _make_graded(
            "w2",
            "Another snippet on same support page.",
            relevance="relevant",
            confidence=0.65,
            source_type="web",
            url="https://support.example.com/warranty/",
        ),
        _make_graded(
            "w3",
            "Unrelated smartphone news.",
            relevance="irrelevant",
            confidence=0.90,
            source_type="web",
            url="https://news.example.com/phone",
        ),
    ]

    result = filter_evidence(internal_chunks=internal, web_chunks=web, min_confidence=0.45)
    assert result.internal_accepted_count == 1
    assert result.web_accepted_count == 1
    assert len(result.accepted) == 2
    assert result.accepted[0].chunk_id == "c1"
    assert result.accepted[0].page_number == 2
    assert result.accepted[1].chunk_id == "w1"
    assert result.accepted[1].url == "https://support.example.com/warranty"
    assert result.rejected_count == 5

    reasons = [item.rejection_reason for item in result.rejected]
    assert any("Duplicate evidence content" in r for r in reasons)
    assert any("Duplicate web source URL" in r for r in reasons)
    assert any("Low relevance confidence" in r for r in reasons)
    assert any("Irrelevant to query" in r for r in reasons)

