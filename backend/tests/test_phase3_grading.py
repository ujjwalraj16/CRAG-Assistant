"""Phase 3 tests: LLM relevance grading, JSON validation, malformed output handling, and outcome classification."""

import json

from app.ollama_client import OllamaUnavailableError, parse_llm_json
from app.retrieval.relevance_grader import grade_retrieved_chunks
from app.schemas.pipeline import RetrievedChunk


def _sample_chunk(chunk_id: str, text: str, score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        document_name="product_manual.pdf",
        category="Electronics",
        page_number=1,
        chunk_index=0,
        text=text,
        similarity_score=score,
    )


def test_parse_llm_json_handles_fences_and_surrounding_text():
    fenced = 'Sure, here is the evaluation:\n```json\n{"relevance": "relevant", "confidence": 0.9, "reason": "Direct match"}\n```'
    parsed = parse_llm_json(fenced)
    assert parsed["relevance"] == "relevant"
    assert parsed["confidence"] == 0.9


def test_grading_correct_retrieval_even_with_moderate_similarity():
    chunks = [
        _sample_chunk("c1", "AeroBook 14 includes a 2-year battery warranty.", score=0.45),
        _sample_chunk("c2", "Battery replacements are covered under warranty for 24 months.", score=0.42),
    ]

    def mock_llm(prompt: str, system_prompt: str | None, expect_json: bool) -> str:
        return json.dumps(
            {
                "relevance": "RELEVANT",
                "confidence": 0.88,
                "reason": "Explains laptop battery warranty coverage.",
            }
        )

    summary = grade_retrieved_chunks("What is the AeroBook battery warranty?", chunks, llm_fn=mock_llm)
    assert summary.outcome == "CORRECT"
    assert summary.relevant_count == 2
    assert summary.irrelevant_count == 0
    assert summary.ollama_available is True
    assert all(not c.parse_fallback for c in summary.graded_chunks)


def test_grading_incorrect_when_chunks_share_keywords_but_miss_intent():
    chunks = [
        _sample_chunk("c1", "Savings account interest policy applies to banking users.", score=0.89),
        _sample_chunk("c2", "Smartphone camera stabilization manual.", score=0.85),
    ]

    def mock_llm(prompt: str, system_prompt: str | None, expect_json: bool) -> str:
        return json.dumps(
            {
                "relevance": "irrelevant",
                "confidence": 0.92,
                "reason": "Chunk is about unrelated products/policies despite high vector score.",
            }
        )

    summary = grade_retrieved_chunks("What is the AeroBook battery warranty?", chunks, llm_fn=mock_llm)
    assert summary.outcome == "INCORRECT"
    assert summary.relevant_count == 0
    assert summary.irrelevant_count == 2


def test_grading_ambiguous_on_partial_evidence():
    chunks = [
        _sample_chunk("c1", "AeroBook 14 comes with a 65W charger.", score=0.72),
        _sample_chunk("c2", "Warranty claims require a valid purchase receipt.", score=0.68),
        _sample_chunk("c3", "Display brightness reaches 400 nits.", score=0.51),
    ]

    responses = iter(
        [
            '{"relevance": "irrelevant", "confidence": 0.8, "reason": "Mentions charger only."}',
            '{"relevance": "relevant", "confidence": 0.58, "reason": "Mentions warranty claim receipt requirement, not duration."}',
            '{"relevance": "irrelevant", "confidence": 0.9, "reason": "Display specs only."}',
        ]
    )

    summary = grade_retrieved_chunks(
        "How long is the AeroBook 14 battery warranty?",
        chunks,
        llm_fn=lambda p, s, j: next(responses),
    )
    assert summary.outcome == "AMBIGUOUS"
    assert summary.relevant_count == 1
    assert summary.irrelevant_count == 2


def test_malformed_llm_json_handled_safely():
    chunks = [_sample_chunk("c1", "Some text about returns.")]

    def broken_llm(prompt: str, system_prompt: str | None, expect_json: bool) -> str:
        return "I think this might be partially relevant, confidence = high"

    summary = grade_retrieved_chunks("Return window?", chunks, llm_fn=broken_llm)
    assert summary.outcome == "INCORRECT"
    assert summary.graded_chunks[0].relevance == "irrelevant"
    assert summary.graded_chunks[0].confidence == 0.0
    assert summary.graded_chunks[0].parse_fallback is True
    assert summary.ollama_available is True


def test_ollama_unavailable_handled_gracefully():
    chunks = [_sample_chunk("c1", "Some text about returns.")]

    def down_llm(prompt: str, system_prompt: str | None, expect_json: bool) -> str:
        raise OllamaUnavailableError("Connection refused on 127.0.0.1:11434")

    summary = grade_retrieved_chunks("Return window?", chunks, llm_fn=down_llm)
    assert summary.outcome == "INCORRECT"
    assert summary.ollama_available is False
    assert summary.graded_chunks[0].parse_fallback is True
    assert "unavailable" in summary.explanation.lower()


def test_empty_chunk_list_returns_incorrect():
    summary = grade_retrieved_chunks("Any query?", [], llm_fn=lambda p, s, j: "")
    assert summary.outcome == "INCORRECT"
    assert summary.graded_chunks == []

