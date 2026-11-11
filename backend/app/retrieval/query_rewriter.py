"""Lightweight query preprocessing, intent detection, LLM query rewriting, and corrective retrieval."""

from pathlib import Path
import re

from app.config import settings
from app.ingestion.indexer import EmbedFn
from app.ollama_client import (
    LLMCallable,
    OllamaUnavailableError,
    call_ollama,
    parse_llm_json,
)
from app.retrieval.relevance_grader import (
    classify_retrieval_outcome,
    grade_retrieved_chunks,
)
from app.retrieval.retriever import retrieve_chunks
from app.schemas.pipeline import (
    CorrectiveRetrievalAttempt,
    CorrectiveRetrievalResult,
    GradedChunk,
    QueryIntent,
    QueryRewriteResult,
    RetrievalOutcome,
)


WHITESPACE_RE = re.compile(r"\s+")

INTENT_KEYWORDS: list[tuple[QueryIntent, tuple[str, ...]]] = [
    (
        "Warranty and policies",
        (
            "warranty",
            "return",
            "refund",
            "policy",
            "policies",
            "exchange",
            "guarantee",
            "coverage",
            "claim",
            "cancel",
            "cancellation",
            "terms",
            "privacy",
        ),
    ),
    (
        "Product comparison",
        (
            "compare",
            "comparison",
            "versus",
            " vs ",
            "difference",
            "differences",
            "better",
            "which one",
            "between",
        ),
    ),
    (
        "Technical support",
        (
            "error",
            "issue",
            "problem",
            "fix",
            "troubleshoot",
            "broken",
            "not working",
            "won't",
            "cannot",
            "reset",
            "setup",
            "install",
            "update",
            "crash",
            "drain",
            "overheat",
        ),
    ),
    (
        "Product information",
        (
            "price",
            "cost",
            "spec",
            "specs",
            "specification",
            "specifications",
            "feature",
            "features",
            "dimension",
            "weight",
            "battery",
            "storage",
            "memory",
            "camera",
            "model",
            "compatible",
            "compatibility",
            "details",
            "supported",
        ),
    ),
    (
        "General knowledge",
        (
            "what is",
            "who is",
            "why does",
            "how does",
            "explain",
            "define",
            "meaning",
            "history",
        ),
    ),
]

REWRITE_SYSTEM_PROMPT = (
    "You are a search query optimizer for a Corrective Retrieval-Augmented Generation (CRAG) system. "
    "Rewrite the user's question into a clear, focused search query that improves document retrieval "
    "in a vector knowledge base. "
    "Do not answer the question and do not invent product names or facts. "
    "Respond ONLY with valid JSON: "
    '{"rewritten_query": "<improved search query>", '
    '"missing_information": "<missing details if partial evidence was given, else null>", '
    '"reason": "<short explanation of how the rewrite improves retrieval>"}'
)

OUTCOME_RANK: dict[RetrievalOutcome, int] = {
    "INCORRECT": 0,
    "AMBIGUOUS": 1,
    "CORRECT": 2,
}


def preprocess_query(query: str) -> str:
    """Trim and normalize whitespace in a user query while preserving meaning."""
    return WHITESPACE_RE.sub(" ", (query or "").strip())


def detect_query_intent(query: str) -> QueryIntent:
    """Classify query intent using lightweight keyword scoring independent of product category."""
    cleaned = f" {preprocess_query(query).lower()} "
    if not cleaned.strip():
        return "Other"

    best_intent: QueryIntent = "Other"
    best_score = 0

    for intent, keywords in INTENT_KEYWORDS:
        score = sum(1 for kw in keywords if kw in cleaned)
        if score > best_score:
            best_score = score
            best_intent = intent

    return best_intent


def rewrite_query(
    original_query: str,
    outcome: RetrievalOutcome,
    partial_chunks: list[GradedChunk] | None = None,
    model: str | None = None,
    base_url: str | None = None,
    llm_fn: LLMCallable | None = None,
) -> QueryRewriteResult:
    """Use Ollama to reformulate a poorly performing query for corrective retrieval."""
    cleaned = preprocess_query(original_query)
    if not cleaned:
        return QueryRewriteResult(
            original_query=original_query,
            rewritten_query="",
            reason="Empty query cannot be rewritten.",
            rewrite_applied=False,
        )

    partial_notes = ""
    if outcome == "AMBIGUOUS" and partial_chunks:
        relevant_snippets = [
            f"- {chunk.document_name}: {chunk.text[:220]}"
            for chunk in partial_chunks
            if chunk.relevance == "relevant"
        ]
        if relevant_snippets:
            partial_notes = (
                "\nPartially relevant evidence already retrieved:\n"
                + "\n".join(relevant_snippets[:3])
                + "\nIdentify what key information is still missing and focus the rewritten query on that gap.\n"
            )

    prompt = (
        f"Original User Query: {cleaned}\n"
        f"Previous Retrieval Outcome: {outcome}\n"
        f"{partial_notes}\n"
        "Rewrite this query into a sharper, unambiguous retrieval query using key domain concepts "
        "and synonyms likely to appear in documentation. Return JSON only."
    )

    try:
        raw_output = (
            llm_fn(prompt, REWRITE_SYSTEM_PROMPT, True)
            if llm_fn is not None
            else call_ollama(
                prompt=prompt,
                system_prompt=REWRITE_SYSTEM_PROMPT,
                expect_json=True,
                model=model,
                base_url=base_url,
                temperature=0.0,
            )
        )
    except OllamaUnavailableError as exc:
        return QueryRewriteResult(
            original_query=cleaned,
            rewritten_query=cleaned,
            reason=f"Ollama unavailable during query rewriting: {exc}",
            rewrite_applied=False,
            ollama_available=False,
        )

    try:
        parsed = parse_llm_json(raw_output)
        rewritten = preprocess_query(str(parsed.get("rewritten_query") or ""))
        missing = parsed.get("missing_information")
        missing_str = preprocess_query(str(missing)) if missing else None
        reason = preprocess_query(str(parsed.get("reason") or "Reformulated query for clearer semantic matching."))
    except ValueError:
        # Fallback if model returned plain text instead of JSON.
        candidate = preprocess_query(raw_output.strip().strip('"').strip("'"))
        if candidate and len(candidate) <= 200 and "\n" not in raw_output.strip():
            rewritten = candidate
            missing_str = None
            reason = "Used plain-text query rewrite after non-JSON LLM response."
        else:
            return QueryRewriteResult(
                original_query=cleaned,
                rewritten_query=cleaned,
                reason="Malformed LLM rewrite output; retained original query.",
                rewrite_applied=False,
                ollama_available=True,
            )

    if not rewritten or rewritten.lower() == cleaned.lower():
        return QueryRewriteResult(
            original_query=cleaned,
            rewritten_query=cleaned,
            missing_information=missing_str,
            reason="Rewritten query did not differ meaningfully from original query.",
            rewrite_applied=False,
            ollama_available=True,
        )

    return QueryRewriteResult(
        original_query=cleaned,
        rewritten_query=rewritten,
        missing_information=missing_str,
        reason=reason,
        rewrite_applied=True,
        ollama_available=True,
    )


def _merge_graded_chunks(
    existing: list[GradedChunk],
    new_items: list[GradedChunk],
) -> list[GradedChunk]:
    """Deduplicate graded chunks by chunk_id, preferring relevant and higher-confidence hits."""
    by_id: dict[str, GradedChunk] = {}
    for item in existing + new_items:
        current = by_id.get(item.chunk_id)
        if current is None:
            by_id[item.chunk_id] = item
            continue
        item_score = (1 if item.relevance == "relevant" else 0, item.confidence)
        current_score = (1 if current.relevance == "relevant" else 0, current.confidence)
        if item_score > current_score:
            by_id[item.chunk_id] = item
    return list(by_id.values())


def perform_corrective_retrieval(
    query: str,
    top_k: int | None = None,
    category: str | None = None,
    document_ids: list[str] | None = None,
    min_similarity: float | None = None,
    max_correction_attempts: int | None = None,
    embedding_model: str | None = None,
    ollama_model: str | None = None,
    ollama_base_url: str | None = None,
    db_path: Path | None = None,
    index_dir: Path | None = None,
    embed_fn: EmbedFn | None = None,
    grade_llm_fn: LLMCallable | None = None,
    rewrite_llm_fn: LLMCallable | None = None,
) -> CorrectiveRetrievalResult:
    """
    Execute Step 3-5 of CRAG: retrieve, grade, and conditionally rewrite/re-retrieve.

    - CORRECT: Accept retrieved chunks without rewriting.
    - INCORRECT / AMBIGUOUS: Rewrite query and re-retrieve up to max_correction_attempts.
    """
    cleaned_query = preprocess_query(query)
    intent = detect_query_intent(cleaned_query)
    max_attempts = (
        settings.max_correction_attempts
        if max_correction_attempts is None
        else max(0, max_correction_attempts)
    )

    initial_chunks = retrieve_chunks(
        query=cleaned_query,
        top_k=top_k,
        category=category,
        document_ids=document_ids,
        min_similarity=min_similarity,
        embedding_model=embedding_model,
        db_path=db_path,
        index_dir=index_dir,
        embed_fn=embed_fn,
    )
    initial_grading = grade_retrieved_chunks(
        query=cleaned_query,
        chunks=initial_chunks,
        model=ollama_model,
        base_url=ollama_base_url,
        llm_fn=grade_llm_fn,
    )

    combined_graded = list(initial_grading.graded_chunks)
    best_outcome: RetrievalOutcome = initial_grading.outcome
    ollama_available = initial_grading.ollama_available
    attempts: list[CorrectiveRetrievalAttempt] = []
    final_rewritten_query: str | None = None

    if best_outcome == "CORRECT" or max_attempts == 0 or not ollama_available:
        return CorrectiveRetrievalResult(
            original_query=query,
            cleaned_query=cleaned_query,
            intent=intent,
            initial_chunks=initial_chunks,
            initial_grading=initial_grading,
            correction_attempted=False,
            attempts=[],
            final_rewritten_query=None,
            final_outcome=best_outcome,
            combined_graded_chunks=combined_graded,
            ollama_available=ollama_available,
        )

    seen_queries = {cleaned_query.lower()}
    current_query = cleaned_query
    current_outcome: RetrievalOutcome = initial_grading.outcome
    current_graded = initial_grading.graded_chunks

    for attempt_idx in range(1, max_attempts + 1):
        rewrite_result = rewrite_query(
            original_query=current_query,
            outcome=current_outcome,
            partial_chunks=current_graded,
            model=ollama_model,
            base_url=ollama_base_url,
            llm_fn=rewrite_llm_fn,
        )
        if not rewrite_result.ollama_available:
            ollama_available = False
            break
        if not rewrite_result.rewrite_applied:
            break

        rewritten_lower = rewrite_result.rewritten_query.lower()
        if rewritten_lower in seen_queries:
            break
        seen_queries.add(rewritten_lower)

        final_rewritten_query = rewrite_result.rewritten_query
        corrected_chunks = retrieve_chunks(
            query=rewrite_result.rewritten_query,
            top_k=top_k,
            category=category,
            document_ids=document_ids,
            min_similarity=min_similarity,
            embedding_model=embedding_model,
            db_path=db_path,
            index_dir=index_dir,
            embed_fn=embed_fn,
        )
        # Always grade new chunks against the original user query so answers remain on-topic.
        corrected_grading = grade_retrieved_chunks(
            query=cleaned_query,
            chunks=corrected_chunks,
            model=ollama_model,
            base_url=ollama_base_url,
            llm_fn=grade_llm_fn,
        )
        if not corrected_grading.ollama_available:
            ollama_available = False

        attempts.append(
            CorrectiveRetrievalAttempt(
                attempt_number=attempt_idx,
                rewritten_query=rewrite_result.rewritten_query,
                missing_information=rewrite_result.missing_information,
                rewrite_reason=rewrite_result.reason,
                retrieved_count=len(corrected_chunks),
                grading=corrected_grading,
            )
        )

        combined_graded = _merge_graded_chunks(combined_graded, corrected_grading.graded_chunks)
        combined_outcome, _ = classify_retrieval_outcome(combined_graded)

        # Keep the strongest outcome across initial, corrected, and merged evidence.
        if OUTCOME_RANK[corrected_grading.outcome] > OUTCOME_RANK[best_outcome]:
            best_outcome = corrected_grading.outcome
        if OUTCOME_RANK[combined_outcome] > OUTCOME_RANK[best_outcome]:
            best_outcome = combined_outcome

        if best_outcome == "CORRECT" or not ollama_available:
            break

        current_query = rewrite_result.rewritten_query
        current_outcome = corrected_grading.outcome
        current_graded = corrected_grading.graded_chunks

    return CorrectiveRetrievalResult(
        original_query=query,
        cleaned_query=cleaned_query,
        intent=intent,
        initial_chunks=initial_chunks,
        initial_grading=initial_grading,
        correction_attempted=len(attempts) > 0,
        attempts=attempts,
        final_rewritten_query=final_rewritten_query,
        final_outcome=best_outcome,
        combined_graded_chunks=combined_graded,
        ollama_available=ollama_available,
    )

