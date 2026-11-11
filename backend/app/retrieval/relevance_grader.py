"""LLM-based relevance evaluator that checks retrieved content against the user query."""

from pydantic import ValidationError

from app.ollama_client import (
    LLMCallable,
    OllamaUnavailableError,
    call_ollama,
    parse_llm_json,
)
from app.schemas.pipeline import (
    GradedChunk,
    GradingSummary,
    RelevanceGrade,
    RetrievalOutcome,
    RetrievedChunk,
)


GRADING_SYSTEM_PROMPT = (
    "You are a strict retrieval relevance evaluator in a Corrective RAG system. "
    "Evaluate whether the supplied evidence snippet contains facts or details that are "
    "actually useful for answering the user's question. "
    "Do not mark a snippet relevant merely because it shares surface keywords if it is "
    "about a different product, policy, or topic. "
    "Respond ONLY with valid JSON matching: "
    '{"relevance": "relevant" | "irrelevant", "confidence": <float 0.0 to 1.0>, "reason": "<short explanation>"}'
)


def build_grading_prompt(query: str, evidence_text: str, source_name: str = "") -> str:
    """Format the relevance grading prompt for a single chunk or web snippet."""
    source_line = f"Source: {source_name}\n" if source_name else ""
    return (
        f"User Query: {query}\n\n"
        f"{source_line}"
        f"Retrieved Content:\n\"\"\"\n{evidence_text}\n\"\"\"\n\n"
        "Does this retrieved content contain information useful for answering the query? "
        "Return JSON only."
    )


def grade_text_relevance(
    query: str,
    evidence_text: str,
    source_name: str = "",
    model: str | None = None,
    base_url: str | None = None,
    llm_fn: LLMCallable | None = None,
) -> tuple[RelevanceGrade, bool, bool]:
    """
    Grade one piece of evidence with Ollama and validate with Pydantic.

    Returns (grade, parse_fallback_used, ollama_reachable).
    """
    prompt = build_grading_prompt(query=query, evidence_text=evidence_text, source_name=source_name)

    try:
        raw_output = (
            llm_fn(prompt, GRADING_SYSTEM_PROMPT, True)
            if llm_fn is not None
            else call_ollama(
                prompt=prompt,
                system_prompt=GRADING_SYSTEM_PROMPT,
                expect_json=True,
                model=model,
                base_url=base_url,
                temperature=0.0,
            )
        )
    except OllamaUnavailableError as exc:
        fallback = RelevanceGrade(
            relevance="irrelevant",
            confidence=0.0,
            reason=f"Ollama unavailable during relevance grading: {exc}",
        )
        return fallback, True, False

    try:
        parsed_dict = parse_llm_json(raw_output)
        grade = RelevanceGrade.model_validate(parsed_dict)
        return grade, False, True
    except (ValueError, ValidationError):
        # Conservative fallback: never trust unverified evidence when LLM JSON is malformed.
        fallback = RelevanceGrade(
            relevance="irrelevant",
            confidence=0.0,
            reason="Malformed LLM grading JSON; safely treated as irrelevant.",
        )
        return fallback, True, True


def classify_retrieval_outcome(graded_chunks: list[GradedChunk]) -> tuple[RetrievalOutcome, str]:
    """Classify overall retrieval quality into CORRECT, INCORRECT, or AMBIGUOUS."""
    if not graded_chunks:
        return "INCORRECT", "No candidate chunks were retrieved from the internal index."

    relevant = [chunk for chunk in graded_chunks if chunk.relevance == "relevant"]
    total = len(graded_chunks)
    relevant_count = len(relevant)

    if relevant_count == 0:
        return "INCORRECT", f"0 of {total} retrieved chunks were relevant to the query."

    high_confidence = [chunk for chunk in relevant if chunk.confidence >= 0.6]
    max_confidence = max(chunk.confidence for chunk in relevant)
    relevant_ratio = relevant_count / total

    # A single weak relevant hit surrounded by irrelevant chunks is insufficient.
    if relevant_count == 1 and max_confidence < 0.5 and total >= 3:
        return (
            "INCORRECT",
            f"Only 1 low-confidence chunk ({max_confidence:.2f}) out of {total} appeared relevant.",
        )

    # Strong evidence exists if multiple confident hits exist or a dominant high-confidence hit covers the query.
    if len(high_confidence) >= 2 or (max_confidence >= 0.75 and relevant_ratio >= 0.4):
        return (
            "CORRECT",
            f"{relevant_count} of {total} retrieved chunks are relevant with sufficient confidence (max {max_confidence:.2f}).",
        )

    return (
        "AMBIGUOUS",
        f"Partial evidence found: {relevant_count} of {total} chunks relevant (max confidence {max_confidence:.2f}).",
    )


def grade_retrieved_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    model: str | None = None,
    base_url: str | None = None,
    llm_fn: LLMCallable | None = None,
) -> GradingSummary:
    """Evaluate all retrieved internal chunks and compute overall retrieval classification."""
    if not chunks:
        return GradingSummary(
            outcome="INCORRECT",
            graded_chunks=[],
            relevant_count=0,
            irrelevant_count=0,
            avg_confidence=0.0,
            explanation="No internal chunks were retrieved for relevance evaluation.",
            ollama_available=True,
        )

    graded: list[GradedChunk] = []
    ollama_available = True

    for chunk in chunks:
        grade, parse_fallback, reachable = grade_text_relevance(
            query=query,
            evidence_text=chunk.text,
            source_name=chunk.document_name,
            model=model,
            base_url=base_url,
            llm_fn=llm_fn,
        )
        if not reachable:
            ollama_available = False

        graded.append(
            GradedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_name=chunk.document_name,
                category=chunk.category,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                similarity_score=chunk.similarity_score,
                source_type="internal",
                relevance=grade.relevance,
                confidence=grade.confidence,
                reason=grade.reason,
                parse_fallback=parse_fallback,
            )
        )

    outcome, explanation = classify_retrieval_outcome(graded)
    if not ollama_available:
        explanation = f"Ollama service unavailable during grading; {explanation}"

    relevant_count = sum(1 for item in graded if item.relevance == "relevant")
    irrelevant_count = len(graded) - relevant_count
    avg_confidence = round(sum(item.confidence for item in graded) / len(graded), 4)

    return GradingSummary(
        outcome=outcome,
        graded_chunks=graded,
        relevant_count=relevant_count,
        irrelevant_count=irrelevant_count,
        avg_confidence=avg_confidence,
        explanation=explanation,
        ollama_available=ollama_available,
    )

