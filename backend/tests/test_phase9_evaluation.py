"""Phase 9 tests: evaluation metrics and Basic RAG vs CRAG benchmark runner."""

import json
from pathlib import Path

import numpy as np

from app.evaluation.benchmark import run_benchmark
from app.evaluation.metrics import (
    compute_answer_correctness,
    compute_answer_faithfulness,
    retrieval_precision_at_k,
    retrieval_recall_at_k,
)


TERMS = [
    "aerobook",
    "battery",
    "warranty",
    "65w",
    "usb",
    "15-day",
    "return",
    "receipt",
    "finserve",
    "apy",
    "fdic",
    "nova",
    "ip68",
    "50mp",
    "weight",
    "hdmi",
    "mortgage",
    "india",
    "price",
]


def deterministic_embedder(texts: list[str]) -> np.ndarray:
    vectors = np.zeros((len(texts), len(TERMS) + 1), dtype=np.float32)
    for row_idx, text in enumerate(texts):
        lowered = text.lower()
        for col_idx, term in enumerate(TERMS):
            vectors[row_idx, col_idx] = float(lowered.count(term))
        vectors[row_idx, -1] = 0.05
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


def test_precision_recall_and_unanswerable_metrics():
    assert retrieval_precision_at_k(["a.txt", "b.txt", "a.txt"], ["a.txt"], k=3) == 0.6667
    assert retrieval_recall_at_k(["a.txt", "b.txt"], ["a.txt", "c.txt"], k=2) == 0.5
    # Out-of-scope query: filtering out all irrelevant docs gives precision 1.0, failing to filter gives 0.0.
    assert retrieval_precision_at_k([], [], k=3) == 1.0
    assert retrieval_precision_at_k(["unrelated.txt"], [], k=3) == 0.0

    faith = compute_answer_faithfulness(
        query="What is warranty?",
        answer="AeroBook offers a 2-year battery warranty.",
        context_texts=["AeroBook Pro 14 includes a 2-year battery warranty."],
    )
    assert faith >= 0.7

    correct = compute_answer_correctness(
        query="What is warranty?",
        answer="AeroBook Pro 14 includes a 2-year battery warranty with 65W USB-C charger.",
        ground_truth_answer="2-year battery warranty and 65W USB-C charger.",
        expected_facts=["2-year", "65W", "USB-C"],
    )
    assert correct >= 0.8

    refusal_score = compute_answer_correctness(
        query="What is mortgage rate?",
        answer="I do not have enough verified evidence to answer that.",
        ground_truth_answer="Not covered.",
        expected_facts=[],
        abstained=True,
        unanswerable_internally=True,
    )
    assert refusal_score == 1.0


def test_benchmark_runner_produces_real_comparison_and_artifacts(tmp_path: Path):
    results_dir = tmp_path / "results"
    app_db = tmp_path / "app.db"

    def mock_grade(prompt: str, sys_prompt: str | None, expect_json: bool) -> str:
        # Reject chunks when query asks about India price or FinServe mortgages.
        if "in India" in prompt or "mortgage" in prompt:
            return json.dumps({"relevance": "irrelevant", "confidence": 0.9, "reason": "Topic not covered in chunk."})
        return json.dumps({"relevance": "relevant", "confidence": 0.9, "reason": "Directly supports query."})

    def mock_rewrite(prompt: str, sys_prompt: str | None, expect_json: bool) -> str:
        return json.dumps(
            {
                "rewritten_query": "AeroBook Pro 14 warranty specs",
                "missing_information": None,
                "reason": "Clarify query.",
            }
        )

    def mock_generate(prompt: str, sys_prompt: str | None, expect_json: bool) -> str:
        if "battery warranty" in prompt:
            return "AeroBook Pro 14 includes a 2-year battery warranty and 65W USB-C charger [Source 1]."
        if "return window" in prompt:
            return "Opened laptops have a 15-day return window requiring the original receipt and serial number [Source 1]."
        if "FDIC" in prompt:
            return "FinServe High-Yield Savings offers 4.25% APY and FDIC insurance up to $250,000 [Source 1]."
        if "water immersion" in prompt:
            return "The Nova X5 has IP68 water resistance and a 50MP main camera [Source 1]."
        if "Compare the warranty" in prompt:
            return "AeroBook Pro 14 has a 2-year battery warranty while Nova X5 has a 1-year limited warranty [Source 1][Source 2]."
        if "weight and HDMI" in prompt:
            return "AeroBook Pro 14 weighs 1.28 kg and has an HDMI 2.1 port [Source 1]."
        # Simulates Basic RAG receiving ungraded FinServe savings doc on mortgage query and hallucinating!
        return "FinServe offers a 6.5% mortgage rate based on account terms [Source 1]."

    def mock_grounding(prompt: str, sys_prompt: str | None, expect_json: bool) -> str:
        return json.dumps({"status": "Supported", "explanation": "Claims match evidence.", "unsupported_claims": []})

    report = run_benchmark(
        results_dir=results_dir,
        top_k=3,
        web_search_enabled=False,
        use_llm_judge=False,
        persist_to_app_db=True,
        app_db_path=app_db,
        embed_fn=deterministic_embedder,
        grade_llm_fn=mock_grade,
        rewrite_llm_fn=mock_rewrite,
        generate_llm_fn=mock_generate,
        grounding_llm_fn=mock_grounding,
    )

    assert (results_dir / "latest_comparison.json").exists()
    assert (results_dir / "latest_comparison.md").exists()

    crag = report["summary"]["crag"]
    basic = report["summary"]["basic_rag"]
    assert crag["total_queries"] == 8
    assert basic["total_queries"] == 8
    # CRAG filters out irrelevant hits on out-of-scope queries and abstains instead of hallucinating.
    assert crag["retrieval_precision_at_k"] > basic["retrieval_precision_at_k"]
    assert crag["answer_correctness"] > basic["answer_correctness"]
    assert crag["abstention_rate"] > 0.0

