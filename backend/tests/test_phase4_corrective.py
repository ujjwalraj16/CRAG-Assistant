"""Phase 4 tests: query intent detection, query rewriting, and corrective retrieval."""

import json
from pathlib import Path

import numpy as np

from app.database.db import init_db
from app.ingestion.indexer import ingest_document
from app.ollama_client import OllamaUnavailableError
from app.retrieval.query_rewriter import (
    detect_query_intent,
    perform_corrective_retrieval,
    preprocess_query,
    rewrite_query,
)


TERMS = [
    "laptop",
    "aerobook",
    "battery",
    "warranty",
    "years",
    "replacement",
    "coverage",
    "smartphone",
    "camera",
    "charger",
    "power",
    "fix",
    "troubleshoot",
    "compare",
]


def keyword_embedder(texts: list[str]) -> np.ndarray:
    """Simple unit-normalized term frequency embedder for deterministic retrieval tests."""
    vectors = np.zeros((len(texts), len(TERMS) + 1), dtype=np.float32)
    for row_idx, text in enumerate(texts):
        lowered = text.lower()
        for col_idx, term in enumerate(TERMS):
            vectors[row_idx, col_idx] = float(lowered.count(term))
        vectors[row_idx, -1] = 0.05
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


def test_preprocess_and_intent_detection():
    assert preprocess_query("  What   is the   return   policy?  ") == "What is the return policy?"
    assert detect_query_intent("What is the warranty coverage for laptops?") == "Warranty and policies"
    assert detect_query_intent("Compare AeroBook 14 vs ProBook 16") == "Product comparison"
    assert detect_query_intent("How to fix battery drain issue after update?") == "Technical support"
    assert detect_query_intent("What are the camera specifications and weight?") == "Product information"
    assert detect_query_intent("Explain what embeddings represent") == "General knowledge"
    assert detect_query_intent("Hello there") == "Other"


def test_correct_retrieval_skips_rewriting(tmp_path: Path):
    db_path = tmp_path / "test.db"
    index_dir = tmp_path / "faiss"
    init_db(db_path)

    doc = tmp_path / "warranty.txt"
    doc.write_text(
        "AeroBook 14 battery warranty coverage lasts 2 years with free battery replacement.",
        encoding="utf-8",
    )
    ingest_document(doc, category="Laptops", db_path=db_path, index_dir=index_dir, embed_fn=keyword_embedder)

    rewrite_calls = 0

    def mock_grade_llm(prompt: str, sys_prompt: str | None, expect_json: bool) -> str:
        return json.dumps({"relevance": "relevant", "confidence": 0.9, "reason": "Direct warranty details."})

    def mock_rewrite_llm(prompt: str, sys_prompt: str | None, expect_json: bool) -> str:
        nonlocal rewrite_calls
        rewrite_calls += 1
        return "{}"

    result = perform_corrective_retrieval(
        query="AeroBook battery warranty",
        top_k=2,
        max_correction_attempts=1,
        db_path=db_path,
        index_dir=index_dir,
        embed_fn=keyword_embedder,
        grade_llm_fn=mock_grade_llm,
        rewrite_llm_fn=mock_rewrite_llm,
    )
    assert result.final_outcome == "CORRECT"
    assert result.correction_attempted is False
    assert result.attempts == []
    assert rewrite_calls == 0


def test_incorrect_retrieval_triggers_rewrite_and_recovers(tmp_path: Path):
    db_path = tmp_path / "test.db"
    index_dir = tmp_path / "faiss"
    init_db(db_path)

    # Initial query ("power") retrieves distracting display doc; rewritten query matches charger doc.
    display_doc = tmp_path / "display.txt"
    display_doc.write_text(
        "Smartphone screen power saving mode reduces brightness.",
        encoding="utf-8",
    )
    charger_doc = tmp_path / "charger_manual.txt"
    charger_doc.write_text(
        "The AeroBook charger supports 65W USB-C delivery and fast replacement.",
        encoding="utf-8",
    )
    ingest_document(display_doc, category="Smartphones", db_path=db_path, index_dir=index_dir, embed_fn=keyword_embedder)
    ingest_document(charger_doc, category="Laptops", db_path=db_path, index_dir=index_dir, embed_fn=keyword_embedder)

    def mock_grade_llm(prompt: str, sys_prompt: str | None, expect_json: bool) -> str:
        if "65W USB-C" in prompt:
            return json.dumps({"relevance": "relevant", "confidence": 0.9, "reason": "Explains 65W charger specs."})
        return json.dumps({"relevance": "irrelevant", "confidence": 0.9, "reason": "Unrelated smartphone display doc."})

    def mock_rewrite_llm(prompt: str, sys_prompt: str | None, expect_json: bool) -> str:
        return json.dumps(
            {
                "rewritten_query": "AeroBook charger replacement",
                "missing_information": None,
                "reason": "Replace vague term 'power' with specific AeroBook charger terms.",
            }
        )

    result = perform_corrective_retrieval(
        query="power adapter wattage",
        top_k=1,
        max_correction_attempts=1,
        db_path=db_path,
        index_dir=index_dir,
        embed_fn=keyword_embedder,
        grade_llm_fn=mock_grade_llm,
        rewrite_llm_fn=mock_rewrite_llm,
    )
    assert result.initial_grading.outcome == "INCORRECT"
    assert result.correction_attempted is True
    assert result.final_rewritten_query == "AeroBook charger replacement"
    assert len(result.attempts) == 1
    assert result.final_outcome == "CORRECT"
    assert any(c.relevance == "relevant" and "65W USB-C" in c.text for c in result.combined_graded_chunks)


def test_ambiguous_retrieval_identifies_missing_info_and_limits_attempts(tmp_path: Path):
    db_path = tmp_path / "test.db"
    index_dir = tmp_path / "faiss"
    init_db(db_path)

    partial_doc = tmp_path / "partial.txt"
    partial_doc.write_text("AeroBook battery service requires serial number verification.", encoding="utf-8")
    full_doc = tmp_path / "full.txt"
    full_doc.write_text("AeroBook battery warranty coverage is 2 years from purchase.", encoding="utf-8")
    ingest_document(partial_doc, category="Laptops", db_path=db_path, index_dir=index_dir, embed_fn=keyword_embedder)
    ingest_document(full_doc, category="Laptops", db_path=db_path, index_dir=index_dir, embed_fn=keyword_embedder)

    def mock_grade_llm(prompt: str, sys_prompt: str | None, expect_json: bool) -> str:
        if "serial number verification" in prompt:
            return json.dumps({"relevance": "relevant", "confidence": 0.55, "reason": "Partial process detail only."})
        return json.dumps({"relevance": "irrelevant", "confidence": 0.85, "reason": "Does not help."})

    rewrite_prompts: list[str] = []

    def mock_rewrite_llm(prompt: str, sys_prompt: str | None, expect_json: bool) -> str:
        rewrite_prompts.append(prompt)
        return json.dumps(
            {
                "rewritten_query": "AeroBook battery warranty coverage duration years",
                "missing_information": "Warranty duration in years",
                "reason": "Target missing warranty duration.",
            }
        )

    result = perform_corrective_retrieval(
        query="AeroBook battery service",
        top_k=1,
        max_correction_attempts=1,
        db_path=db_path,
        index_dir=index_dir,
        embed_fn=keyword_embedder,
        grade_llm_fn=mock_grade_llm,
        rewrite_llm_fn=mock_rewrite_llm,
    )
    assert result.initial_grading.outcome == "AMBIGUOUS"
    assert result.final_outcome == "AMBIGUOUS"
    assert result.correction_attempted is True
    assert len(result.attempts) == 1
    assert "Partially relevant evidence already retrieved" in rewrite_prompts[0]
    assert result.attempts[0].missing_information == "Warranty duration in years"


def test_rewrite_handles_ollama_unavailable_and_malformed_json():
    down = rewrite_query(
        "laptop warranty",
        outcome="INCORRECT",
        llm_fn=lambda p, s, j: (_ for _ in ()).throw(OllamaUnavailableError("down")),
    )
    assert down.ollama_available is False
    assert down.rewrite_applied is False
    assert down.rewritten_query == "laptop warranty"

    malformed = rewrite_query(
        "laptop warranty",
        outcome="INCORRECT",
        llm_fn=lambda p, s, j: "Here is some invalid\nmulti-line\nprose response",
    )
    assert malformed.ollama_available is True
    assert malformed.rewrite_applied is False
    assert malformed.rewritten_query == "laptop warranty"
