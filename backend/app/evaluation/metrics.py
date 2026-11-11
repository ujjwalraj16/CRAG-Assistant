"""RAGAS-compatible retrieval and generation evaluation metrics."""

import re
from typing import Any

from app.ollama_client import LLMCallable, OllamaUnavailableError, call_ollama, parse_llm_json


TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "its", "of", "on", "or", "source", "that", "the", "to", "was",
    "were", "will", "with", "this", "these", "those", "their", "them",
}
REFUSAL_PHRASES = (
    "do not have enough verified evidence",
    "does not specify",
    "do not specify",
    "not provided",
    "not mentioned",
    "not available",
    "unavailable",
    "does not cover",
    "do not cover",
    "no information",
    "cannot find",
    "cannot verify",
)


def _content_tokens(text: str) -> set[str]:
    tokens = TOKEN_RE.findall(text.lower())
    return {tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1}


def retrieval_precision_at_k(
    context_docs: list[str],
    relevant_docs: list[str],
    k: int | None = None,
) -> float:
    """Compute Precision@K over retrieved/accepted contexts against ground-truth document names."""
    window = context_docs[:k] if k is not None else context_docs
    relevant_set = set(relevant_docs)

    if not relevant_set:
        # For unanswerable/out-of-scope queries, filtering out all irrelevant docs achieves precision 1.0.
        return 1.0 if len(window) == 0 else 0.0

    if not window:
        return 0.0

    hits = sum(1 for doc in window if doc in relevant_set)
    return round(hits / len(window), 4)


def retrieval_recall_at_k(
    context_docs: list[str],
    relevant_docs: list[str],
    k: int | None = None,
) -> float:
    """Compute Recall@K measuring fraction of ground-truth documents surfaced in top-k contexts."""
    relevant_set = set(relevant_docs)
    if not relevant_set:
        return 1.0

    window = set(context_docs[:k] if k is not None else context_docs)
    hits = len(window & relevant_set)
    return round(hits / len(relevant_set), 4)


def compute_context_relevance(
    context_docs: list[str],
    relevant_docs: list[str],
) -> float:
    """Measure proportion of final generator context coming from ground-truth relevant documents."""
    return retrieval_precision_at_k(context_docs, relevant_docs, k=None)


def _llm_judge_score(
    prompt: str,
    model: str | None = None,
    base_url: str | None = None,
    llm_fn: LLMCallable | None = None,
) -> float | None:
    system_prompt = (
        "You are a strict RAG evaluation judge. Return ONLY JSON: "
        '{"score": <float between 0.0 and 1.0>, "reason": "<short reason>"}'
    )
    try:
        raw = (
            llm_fn(prompt, system_prompt, True)
            if llm_fn is not None
            else call_ollama(
                prompt=prompt,
                system_prompt=system_prompt,
                expect_json=True,
                model=model,
                base_url=base_url,
                temperature=0.0,
            )
        )
        parsed = parse_llm_json(raw)
        score = float(parsed["score"])
        return max(0.0, min(1.0, round(score, 4)))
    except (OllamaUnavailableError, ValueError, KeyError, TypeError):
        return None


def compute_answer_faithfulness(
    query: str,
    answer: str,
    context_texts: list[str],
    abstained: bool = False,
    use_llm_judge: bool = False,
    model: str | None = None,
    base_url: str | None = None,
    llm_fn: LLMCallable | None = None,
) -> float:
    """Measure whether claims in the generated answer are supported by supplied evidence."""
    if abstained:
        return 1.0
    if not context_texts or not answer.strip():
        return 0.0

    if use_llm_judge:
        joined_context = "\n---\n".join(context_texts)
        prompt = (
            f"Question: {query}\n\n"
            f"Context:\n{joined_context}\n\n"
            f"Answer:\n{answer}\n\n"
            "Score how faithfully the Answer is grounded ONLY in Context (0.0 = fabricated, 1.0 = completely grounded)."
        )
        judged = _llm_judge_score(prompt, model=model, base_url=base_url, llm_fn=llm_fn)
        if judged is not None:
            return judged

    # Deterministic lexical grounding fallback.
    answer_tokens = _content_tokens(answer)
    if not answer_tokens:
        return 0.0
    context_tokens = _content_tokens(" ".join(context_texts))
    overlap = len(answer_tokens & context_tokens)
    return round(min(1.0, overlap / len(answer_tokens)), 4)


def compute_answer_correctness(
    query: str,
    answer: str,
    ground_truth_answer: str,
    expected_facts: list[str],
    abstained: bool = False,
    unanswerable_internally: bool = False,
    use_llm_judge: bool = False,
    model: str | None = None,
    base_url: str | None = None,
    llm_fn: LLMCallable | None = None,
) -> float:
    """Measure factual correctness of the answer against reference answer and required facts."""
    lowered = answer.lower()

    if unanswerable_internally:
        if abstained or any(phrase in lowered for phrase in REFUSAL_PHRASES):
            return 1.0
        if use_llm_judge:
            prompt = (
                f"Question: {query}\n"
                f"Reference Answer (expected abstention/limitation): {ground_truth_answer}\n"
                f"Candidate Answer: {answer}\n\n"
                "Score 1.0 if Candidate Answer honestly acknowledges missing information rather than inventing facts, else 0.0."
            )
            judged = _llm_judge_score(prompt, model=model, base_url=base_url, llm_fn=llm_fn)
            if judged is not None:
                return judged
        return 0.0

    if abstained or not answer.strip():
        return 0.0

    if use_llm_judge:
        prompt = (
            f"Question: {query}\n"
            f"Reference Answer: {ground_truth_answer}\n"
            f"Candidate Answer: {answer}\n\n"
            "Score factual correctness of Candidate Answer relative to Reference Answer from 0.0 to 1.0."
        )
        judged = _llm_judge_score(prompt, model=model, base_url=base_url, llm_fn=llm_fn)
        if judged is not None:
            return judged

    # Deterministic fact recall + token F1 fallback.
    if expected_facts:
        fact_hits = sum(1 for fact in expected_facts if fact.lower() in lowered)
        fact_score = fact_hits / len(expected_facts)
    else:
        fact_score = 0.0

    ans_tokens = _content_tokens(answer)
    ref_tokens = _content_tokens(ground_truth_answer)
    if ans_tokens and ref_tokens:
        common = len(ans_tokens & ref_tokens)
        precision = common / len(ans_tokens)
        recall = common / len(ref_tokens)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    else:
        f1 = 0.0

    if expected_facts:
        return round(0.75 * fact_score + 0.25 * f1, 4)
    return round(f1, 4)


def aggregate_mode_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute average benchmark metrics across all evaluated dataset queries for one mode."""
    if not records:
        return {
            "total_queries": 0,
            "retrieval_precision_at_k": 0.0,
            "retrieval_recall_at_k": 0.0,
            "context_relevance": 0.0,
            "answer_faithfulness": 0.0,
            "answer_correctness": 0.0,
            "avg_latency_ms": 0.0,
            "correction_frequency": 0.0,
            "web_fallback_frequency": 0.0,
            "abstention_rate": 0.0,
        }

    n = len(records)
    return {
        "total_queries": n,
        "retrieval_precision_at_k": round(sum(r["precision_at_k"] for r in records) / n, 4),
        "retrieval_recall_at_k": round(sum(r["recall_at_k"] for r in records) / n, 4),
        "context_relevance": round(sum(r["context_relevance"] for r in records) / n, 4),
        "answer_faithfulness": round(sum(r["answer_faithfulness"] for r in records) / n, 4),
        "answer_correctness": round(sum(r["answer_correctness"] for r in records) / n, 4),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in records) / n, 2),
        "correction_frequency": round(
            sum(1 for r in records if r.get("correction_attempted")) / n, 4
        ),
        "web_fallback_frequency": round(
            sum(1 for r in records if r.get("web_fallback_used")) / n, 4
        ),
        "abstention_rate": round(sum(1 for r in records if r.get("abstained")) / n, 4),
    }

