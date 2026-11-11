"""Phase 6 tests: answer generation, grounding verification, regeneration/abstention, and end-to-end CRAG pipeline."""

import json
from pathlib import Path

import numpy as np

from app.crag.generator import generate_answer
from app.crag.grounding_checker import verify_grounding
from app.crag.pipeline import run_basic_rag_pipeline, run_crag_pipeline
from app.crag.trace_manager import load_trace
from app.database.db import init_db
from app.ingestion.indexer import ingest_document
from app.ollama_client import OllamaUnavailableError
from app.schemas.pipeline import GradedChunk, WebSearchRawResult


TERMS = [
    "aerobook",
    "laptop",
    "battery",
    "warranty",
    "years",
    "hdmi",
    "charger",
    "usb",
    "price",
    "refund",
    "policy",
]


def keyword_embedder(texts: list[str]) -> np.ndarray:
    vectors = np.zeros((len(texts), len(TERMS) + 1), dtype=np.float32)
    for row_idx, text in enumerate(texts):
        lowered = text.lower()
        for col_idx, term in enumerate(TERMS):
            vectors[row_idx, col_idx] = float(lowered.count(term))
        vectors[row_idx, -1] = 0.05
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


def _graded_evidence(text: str = "AeroBook 14 comes with a 2-year battery warranty.") -> list[GradedChunk]:
    return [
        GradedChunk(
            chunk_id="c1",
            document_id="doc-1",
            document_name="warranty.pdf",
            category="Laptops",
            page_number=3,
            chunk_index=0,
            text=text,
            similarity_score=0.89,
            source_type="internal",
            relevance="relevant",
            confidence=0.92,
            reason="Directly states warranty duration.",
        )
    ]


def test_generator_abstains_when_no_evidence_supplied():
    result = generate_answer("What is the warranty period?", evidence=[])
    assert result.abstained is True
    assert result.citations == []
    assert "do not have enough verified evidence" in result.answer.lower()


def test_generator_handles_ollama_unavailable_gracefully():
    evidence = _graded_evidence()
    result = generate_answer(
        "What is the warranty period?",
        evidence=evidence,
        llm_fn=lambda p, s, j: (_ for _ in ()).throw(OllamaUnavailableError("connection refused")),
    )
    assert result.abstained is True
    assert result.ollama_available is False
    assert len(result.citations) == 1


def test_grounding_checker_verdicts_and_malformed_json():
    evidence = _graded_evidence()

    supported = verify_grounding(
        "What is the battery warranty?",
        "AeroBook 14 has a 2-year battery warranty [Source 1].",
        evidence,
        llm_fn=lambda p, s, j: json.dumps(
            {
                "status": "supported",
                "explanation": "Matches the 2-year warranty stated in Source 1.",
                "unsupported_claims": [],
            }
        ),
    )
    assert supported.status == "Supported"
    assert supported.unsupported_claims == []

    malformed = verify_grounding(
        "What is the battery warranty?",
        "AeroBook 14 has a 2-year battery warranty.",
        evidence,
        llm_fn=lambda p, s, j: "Not valid JSON at all",
    )
    assert malformed.status == "Partially Supported"
    assert malformed.parse_fallback is True


def test_crag_pipeline_happy_path_and_trace_persistence(tmp_path: Path):
    db_path = tmp_path / "test.db"
    index_dir = tmp_path / "faiss"
    init_db(db_path)

    doc = tmp_path / "laptop_warranty.txt"
    doc.write_text(
        "AeroBook 14 includes a 2-year battery warranty and 65W USB charger.",
        encoding="utf-8",
    )
    ingest_document(doc, category="Laptops", db_path=db_path, index_dir=index_dir, embed_fn=keyword_embedder)

    trace = run_crag_pipeline(
        query="What is the AeroBook battery warranty?",
        db_path=db_path,
        index_dir=index_dir,
        embed_fn=keyword_embedder,
        grade_llm_fn=lambda p, s, j: json.dumps(
            {"relevance": "relevant", "confidence": 0.91, "reason": "Direct warranty match."}
        ),
        generate_llm_fn=lambda p, s, j: "The AeroBook 14 includes a 2-year battery warranty [Source 1].",
        grounding_llm_fn=lambda p, s, j: json.dumps(
            {"status": "Supported", "explanation": "Fully grounded in Source 1.", "unsupported_claims": []}
        ),
    )

    assert trace.internal_outcome == "CORRECT"
    assert trace.abstained is False
    assert trace.grounding_status == "Supported"
    assert len(trace.stages) == 8
    assert trace.stages[3].status == "Skipped"  # Query Correction skipped
    assert trace.stages[4].status == "Skipped"  # External Web Retrieval skipped
    assert trace.stages[7].status == "Completed"

    loaded = load_trace(trace.query_id, db_path=db_path)
    assert loaded is not None
    assert loaded.query_id == trace.query_id
    assert loaded.answer == trace.answer


def test_crag_pipeline_regenerates_once_then_abstains_if_still_unsupported(tmp_path: Path):
    db_path = tmp_path / "test.db"
    index_dir = tmp_path / "faiss"
    init_db(db_path)

    doc = tmp_path / "specs.txt"
    doc.write_text("AeroBook 14 includes a 2-year battery warranty.", encoding="utf-8")
    ingest_document(doc, category="Laptops", db_path=db_path, index_dir=index_dir, embed_fn=keyword_embedder)

    gen_calls = 0

    def mock_generate(p: str, s: str | None, j: bool) -> str:
        nonlocal gen_calls
        gen_calls += 1
        return "AeroBook 14 costs $999 and includes free headphones."

    trace = run_crag_pipeline(
        query="What is the AeroBook battery warranty?",
        db_path=db_path,
        index_dir=index_dir,
        embed_fn=keyword_embedder,
        grade_llm_fn=lambda p, s, j: json.dumps(
            {"relevance": "relevant", "confidence": 0.9, "reason": "Relevant warranty text."}
        ),
        generate_llm_fn=mock_generate,
        grounding_llm_fn=lambda p, s, j: json.dumps(
            {
                "status": "Unsupported",
                "explanation": "Mentions price and headphones not found in evidence.",
                "unsupported_claims": ["Costs $999", "Includes free headphones"],
            }
        ),
    )
    assert gen_calls == 2  # Initial generation + 1 regeneration attempt
    assert trace.regeneration_attempted is True
    assert trace.abstained is True
    assert trace.grounding_status == "Abstained"
    assert "do not have enough verified evidence" in trace.answer.lower()


def test_crag_pipeline_uses_web_fallback_when_internal_insufficient(tmp_path: Path):
    db_path = tmp_path / "test.db"
    index_dir = tmp_path / "faiss"
    init_db(db_path)

    doc = tmp_path / "refund.txt"
    doc.write_text("Standard refund policy allows returns within 30 days.", encoding="utf-8")
    ingest_document(doc, category="Policies", db_path=db_path, index_dir=index_dir, embed_fn=keyword_embedder)

    def mock_search(query: str, limit: int, api_key: str) -> list[WebSearchRawResult]:
        return [
            WebSearchRawResult(
                title="AeroBook HDMI Guide",
                url="https://example.com/aerobook-hdmi",
                content="AeroBook 14 features an HDMI 2.1 output port.",
                score=0.95,
            )
        ]

    def mock_grade(p: str, s: str | None, j: bool) -> str:
        if "HDMI 2.1" in p:
            return json.dumps({"relevance": "relevant", "confidence": 0.9, "reason": "Answers HDMI question."})
        return json.dumps({"relevance": "irrelevant", "confidence": 0.85, "reason": "Refund policy only."})

    trace = run_crag_pipeline(
        query="Does AeroBook have HDMI 2.1?",
        web_search_enabled=True,
        db_path=db_path,
        index_dir=index_dir,
        embed_fn=keyword_embedder,
        grade_llm_fn=mock_grade,
        rewrite_llm_fn=lambda p, s, j: json.dumps(
            {
                "rewritten_query": "AeroBook HDMI port specs",
                "missing_information": None,
                "reason": "Clarify HDMI specification.",
            }
        ),
        search_fn=mock_search,
        generate_llm_fn=lambda p, s, j: "Yes, AeroBook 14 features an HDMI 2.1 output port [Source 1].",
        grounding_llm_fn=lambda p, s, j: json.dumps(
            {"status": "Supported", "explanation": "Supported by web source.", "unsupported_claims": []}
        ),
    )

    assert trace.web_outcome is not None
    assert trace.web_outcome.status == "SUCCESS"
    assert len(trace.accepted_evidence) == 1
    assert trace.accepted_evidence[0].source_type == "web"
    assert trace.citations[0].url == "https://example.com/aerobook-hdmi"
    assert trace.grounding_status == "Supported"


def test_basic_rag_pipeline_runs_without_grading_or_web(tmp_path: Path):
    db_path = tmp_path / "test.db"
    index_dir = tmp_path / "faiss"
    init_db(db_path)

    doc = tmp_path / "refund.txt"
    doc.write_text("Standard refund policy allows returns within 30 days.", encoding="utf-8")
    ingest_document(doc, category="Policies", db_path=db_path, index_dir=index_dir, embed_fn=keyword_embedder)

    trace = run_basic_rag_pipeline(
        query="What is the refund policy?",
        db_path=db_path,
        index_dir=index_dir,
        embed_fn=keyword_embedder,
        generate_llm_fn=lambda p, s, j: "Returns are allowed within 30 days [Source 1].",
    )
    assert trace.mode == "basic_rag"
    assert trace.grounding_status == "Skipped"
    assert trace.stages[2].status == "Skipped"  # Relevance Evaluation skipped
    assert trace.stages[4].status == "Skipped"  # External Web Retrieval skipped
    assert "30 days" in trace.answer

