"""Main CRAG and Basic RAG orchestration pipelines."""

from pathlib import Path
import time
import uuid

from app.config import get_public_settings, settings
from app.crag.generator import build_abstention_message, generate_answer
from app.crag.grounding_checker import verify_grounding
from app.crag.trace_manager import TraceBuilder, persist_trace
from app.database.models import get_all_setting_overrides, utc_now_iso
from app.ingestion.indexer import EmbedFn
from app.ollama_client import LLMCallable
from app.retrieval.evidence_filter import filter_evidence
from app.retrieval.query_rewriter import (
    detect_query_intent,
    perform_corrective_retrieval,
    preprocess_query,
)
from app.retrieval.retriever import retrieve_chunks
from app.retrieval.web_search import WebSearchFn, run_web_fallback
from app.schemas.pipeline import (
    GradedChunk,
    PipelineTrace,
    TraceGroundingStatus,
    WebSearchOutcome,
)


def _elapsed_ms(start_perf: float) -> float:
    return (time.perf_counter() - start_perf) * 1000.0


def _resolve_runtime_config(db_path: Path | None = None) -> dict:
    try:
        overrides = get_all_setting_overrides(db_path=db_path)
    except Exception:
        overrides = {}
    return get_public_settings(overrides)


def run_crag_pipeline(
    query: str,
    conversation_id: str | None = None,
    category: str | None = None,
    document_ids: list[str] | None = None,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
    max_correction_attempts: int | None = None,
    web_search_enabled: bool | None = None,
    min_evidence_confidence: float = 0.45,
    persist_trace_record: bool = True,
    db_path: Path | None = None,
    index_dir: Path | None = None,
    embed_fn: EmbedFn | None = None,
    grade_llm_fn: LLMCallable | None = None,
    rewrite_llm_fn: LLMCallable | None = None,
    generate_llm_fn: LLMCallable | None = None,
    grounding_llm_fn: LLMCallable | None = None,
    search_fn: WebSearchFn | None = None,
) -> PipelineTrace:
    """
    Execute the complete 8-stage Adaptive Corrective RAG pipeline:

    1. Query Analysis (preprocessing + lightweight intent detection)
    2. Document Retrieval (FAISS cosine similarity search)
    3. Relevance Evaluation (LLM chunk grading + CORRECT/INCORRECT/AMBIGUOUS)
    4. Query Correction (bounded rewriting + re-retrieval when needed)
    5. External Web Retrieval (Tavily fallback + grading when internal evidence insufficient)
    6. Evidence Filtering (keep high-confidence unique internal/web evidence)
    7. Answer Generation (strictly grounded answer with citations)
    8. Grounding Verification (check claims, regenerate once if unsupported, else abstain)
    """
    pipeline_start = time.perf_counter()
    runtime_cfg = _resolve_runtime_config(db_path=db_path)

    effective_top_k = top_k if top_k is not None else runtime_cfg["retrieval_top_k"]
    effective_similarity = (
        similarity_threshold
        if similarity_threshold is not None
        else runtime_cfg["similarity_threshold"]
    )
    effective_max_corrections = (
        max_correction_attempts
        if max_correction_attempts is not None
        else runtime_cfg["max_correction_attempts"]
    )
    effective_web_enabled = (
        web_search_enabled
        if web_search_enabled is not None
        else runtime_cfg["web_search_enabled"]
    )
    ollama_model = runtime_cfg["ollama_model"]
    ollama_base_url = runtime_cfg["ollama_base_url"]
    embedding_model = runtime_cfg["embedding_model"]

    query_id = str(uuid.uuid4())
    tracer = TraceBuilder(
        query_id=query_id,
        original_query=query,
        mode="crag",
        conversation_id=conversation_id,
    )

    # ------------------------------------------------------------------
    # Stage 1: Query Analysis
    # ------------------------------------------------------------------
    s1 = time.perf_counter()
    cleaned_query = preprocess_query(query)
    intent = detect_query_intent(cleaned_query)
    if not cleaned_query:
        tracer.record_stage(
            1,
            "Query Analysis",
            "Failed",
            _elapsed_ms(s1),
            "Query is empty after preprocessing.",
            {"intent": intent},
        )
        for num, name in [
            (2, "Document Retrieval"),
            (3, "Relevance Evaluation"),
            (4, "Query Correction"),
            (5, "External Web Retrieval"),
            (6, "Evidence Filtering"),
            (7, "Answer Generation"),
            (8, "Grounding Verification"),
        ]:
            tracer.record_stage(num, name, "Skipped", 0.0, "Skipped due to empty query.")
        trace = PipelineTrace(
            query_id=query_id,
            conversation_id=conversation_id,
            mode="crag",
            original_query=query,
            cleaned_query="",
            intent=intent,
            answer="Please enter a non-empty question so I can search the knowledge base.",
            abstained=True,
            grounding_status="Abstained",
            stages=tracer.stages,
            stage_timings_ms=tracer.stage_timings_ms,
            total_latency_ms=round(_elapsed_ms(pipeline_start), 2),
            created_at=utc_now_iso(),
        )
        if persist_trace_record:
            persist_trace(trace, db_path=db_path)
        return trace

    tracer.record_stage(
        1,
        "Query Analysis",
        "Completed",
        _elapsed_ms(s1),
        f"Cleaned query and classified lightweight intent as '{intent}'.",
        {"cleaned_query": cleaned_query, "intent": intent, "category_filter": category or "All"},
    )

    # ------------------------------------------------------------------
    # Stages 2, 3, 4: Internal Retrieval, Grading, and Query Correction
    # ------------------------------------------------------------------
    s2 = time.perf_counter()
    corrective_result = perform_corrective_retrieval(
        query=cleaned_query,
        top_k=effective_top_k,
        category=category,
        document_ids=document_ids,
        min_similarity=effective_similarity,
        max_correction_attempts=effective_max_corrections,
        embedding_model=embedding_model,
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url,
        db_path=db_path,
        index_dir=index_dir,
        embed_fn=embed_fn,
        grade_llm_fn=grade_llm_fn,
        rewrite_llm_fn=rewrite_llm_fn,
    )
    total_retrieval_block_ms = _elapsed_ms(s2)

    initial_chunks = corrective_result.initial_chunks
    initial_grading = corrective_result.initial_grading

    # Estimate stage split cleanly while preserving observed block timing.
    n_parts = 2 + len(corrective_result.attempts)
    per_part_ms = total_retrieval_block_ms / max(n_parts, 1)

    tracer.record_stage(
        2,
        "Document Retrieval",
        "Completed",
        per_part_ms,
        (
            f"Retrieved {len(initial_chunks)} internal chunk(s) from FAISS."
            if initial_chunks
            else "No internal chunks met similarity threshold or index is empty."
        ),
        {
            "retrieved_count": len(initial_chunks),
            "top_similarity": initial_chunks[0].similarity_score if initial_chunks else 0.0,
            "similarity_scores": [c.similarity_score for c in initial_chunks],
            "documents": sorted({c.document_name for c in initial_chunks}),
        },
    )

    if not initial_chunks:
        s3_status = "Skipped"
        s3_explanation = "No initial chunks retrieved to grade; classified as INCORRECT."
    elif not initial_grading.ollama_available:
        s3_status = "Failed"
        s3_explanation = initial_grading.explanation
    else:
        s3_status = "Completed"
        s3_explanation = initial_grading.explanation

    tracer.record_stage(
        3,
        "Relevance Evaluation",
        s3_status,
        per_part_ms,
        s3_explanation,
        {
            "outcome": initial_grading.outcome,
            "relevant_count": initial_grading.relevant_count,
            "irrelevant_count": initial_grading.irrelevant_count,
            "avg_confidence": initial_grading.avg_confidence,
        },
    )

    if corrective_result.correction_attempted:
        s4_status = "Completed" if corrective_result.ollama_available else "Failed"
        s4_explanation = (
            f"Rewrote query to '{corrective_result.final_rewritten_query}'; "
            f"updated outcome: {corrective_result.final_outcome}."
        )
    elif initial_grading.outcome == "CORRECT":
        s4_status = "Skipped"
        s4_explanation = "Initial retrieval rated CORRECT; query correction skipped."
    elif not initial_grading.ollama_available:
        s4_status = "Skipped"
        s4_explanation = "Ollama unavailable during grading; skipped query correction."
    elif effective_max_corrections == 0:
        s4_status = "Skipped"
        s4_explanation = "Max correction attempts configured to 0."
    else:
        s4_status = "Skipped"
        s4_explanation = "Query rewrite produced no alternative search terms."

    tracer.record_stage(
        4,
        "Query Correction",
        s4_status,
        per_part_ms * len(corrective_result.attempts) if corrective_result.attempts else 0.0,
        s4_explanation,
        {
            "correction_attempted": corrective_result.correction_attempted,
            "rewritten_query": corrective_result.final_rewritten_query,
            "attempts": [
                {
                    "attempt": a.attempt_number,
                    "rewritten_query": a.rewritten_query,
                    "missing_information": a.missing_information,
                    "reason": a.rewrite_reason,
                    "retrieved_count": a.retrieved_count,
                    "outcome": a.grading.outcome,
                }
                for a in corrective_result.attempts
            ],
            "final_outcome": corrective_result.final_outcome,
        },
    )

    # ------------------------------------------------------------------
    # Stage 5: External Web Retrieval
    # ------------------------------------------------------------------
    s5 = time.perf_counter()
    web_outcome: WebSearchOutcome
    if corrective_result.final_outcome == "CORRECT":
        web_outcome = WebSearchOutcome(
            status="SKIPPED",
            executed=False,
            explanation="Internal evidence is CORRECT; external web search not required.",
        )
        tracer.record_stage(
            5,
            "External Web Retrieval",
            "Skipped",
            _elapsed_ms(s5),
            web_outcome.explanation,
            {"status": web_outcome.status, "executed": False},
        )
    elif not corrective_result.ollama_available:
        web_outcome = WebSearchOutcome(
            status="SKIPPED",
            executed=False,
            explanation="Skipped web grading because Ollama service is unavailable.",
        )
        tracer.record_stage(
            5,
            "External Web Retrieval",
            "Skipped",
            _elapsed_ms(s5),
            web_outcome.explanation,
            {"status": web_outcome.status, "executed": False},
        )
    else:
        web_outcome = run_web_fallback(
            query=cleaned_query,
            web_search_enabled=effective_web_enabled,
            tavily_api_key=settings.tavily_api_key,
            max_results=settings.web_search_max_results,
            ollama_model=ollama_model,
            ollama_base_url=ollama_base_url,
            search_fn=search_fn,
            grade_llm_fn=grade_llm_fn,
        )
        if web_outcome.status in {"SKIPPED", "UNAVAILABLE"}:
            s5_status = "Skipped"
        elif web_outcome.status == "ERROR":
            s5_status = "Failed"
        else:
            s5_status = "Completed"

        tracer.record_stage(
            5,
            "External Web Retrieval",
            s5_status,
            _elapsed_ms(s5),
            web_outcome.explanation,
            {
                "status": web_outcome.status,
                "executed": web_outcome.executed,
                "query_used": web_outcome.query_used,
                "raw_results_count": web_outcome.raw_results_count,
                "relevant_count": web_outcome.relevant_count,
                "error": web_outcome.error,
            },
        )

    # ------------------------------------------------------------------
    # Stage 6: Evidence Filtering
    # ------------------------------------------------------------------
    s6 = time.perf_counter()
    filter_result = filter_evidence(
        internal_chunks=corrective_result.combined_graded_chunks,
        web_chunks=web_outcome.graded_results,
        min_confidence=min_evidence_confidence,
    )
    tracer.record_stage(
        6,
        "Evidence Filtering",
        "Completed",
        _elapsed_ms(s6),
        (
            f"Accepted {len(filter_result.accepted)} evidence chunk(s) "
            f"({filter_result.internal_accepted_count} internal, "
            f"{filter_result.web_accepted_count} web); "
            f"rejected {filter_result.rejected_count}."
        ),
        {
            "accepted_count": len(filter_result.accepted),
            "internal_accepted": filter_result.internal_accepted_count,
            "web_accepted": filter_result.web_accepted_count,
            "rejected_count": filter_result.rejected_count,
        },
    )

    # ------------------------------------------------------------------
    # Stage 7: Answer Generation
    # ------------------------------------------------------------------
    s7 = time.perf_counter()
    ollama_healthy = corrective_result.ollama_available and web_outcome.ollama_available

    if not ollama_healthy:
        generation_result = generate_answer(
            query=cleaned_query,
            evidence=[],
            model=ollama_model,
            base_url=ollama_base_url,
        )
        generation_result.answer = (
            "The local Ollama LLM service is currently unreachable, so CRAG could not "
            "evaluate relevance or synthesize a grounded response. Please verify Ollama is "
            f"running (`ollama serve`) at `{ollama_base_url}` with model `{ollama_model}`."
        )
        generation_result.ollama_available = False
        generation_result.abstained = True
        tracer.record_stage(
            7,
            "Answer Generation",
            "Failed",
            _elapsed_ms(s7),
            "Skipped generation because local Ollama service is unreachable.",
            {"abstained": True, "ollama_available": False},
        )
    else:
        generation_result = generate_answer(
            query=cleaned_query,
            evidence=filter_result.accepted,
            model=ollama_model,
            base_url=ollama_base_url,
            llm_fn=generate_llm_fn,
        )
        s7_status = "Failed" if not generation_result.ollama_available else "Completed"
        s7_explanation = (
            "Abstained because no verified evidence survived filtering."
            if generation_result.abstained and generation_result.ollama_available
            else (
                "Ollama unreachable during answer generation."
                if not generation_result.ollama_available
                else f"Generated cited answer using {len(generation_result.citations)} verified source(s)."
            )
        )
        tracer.record_stage(
            7,
            "Answer Generation",
            s7_status,
            _elapsed_ms(s7),
            s7_explanation,
            {
                "abstained": generation_result.abstained,
                "citations_count": len(generation_result.citations),
                "ollama_available": generation_result.ollama_available,
            },
        )

    # ------------------------------------------------------------------
    # Stage 8: Grounding Verification (+ optional single regeneration)
    # ------------------------------------------------------------------
    s8 = time.perf_counter()
    grounding_result = None
    regeneration_attempted = False
    final_grounding_status: TraceGroundingStatus = "Skipped"

    if generation_result.abstained or not generation_result.ollama_available:
        reason = (
            "Grounding verification skipped because Ollama is unavailable."
            if not generation_result.ollama_available
            else "Grounding verification skipped because the assistant abstained due to insufficient evidence."
        )
        final_grounding_status = "Abstained" if generation_result.abstained and generation_result.ollama_available else "Skipped"
        tracer.record_stage(
            8,
            "Grounding Verification",
            "Skipped",
            _elapsed_ms(s8),
            reason,
            {"status": final_grounding_status},
        )
    else:
        grounding_result = verify_grounding(
            query=cleaned_query,
            answer=generation_result.answer,
            evidence=filter_result.accepted,
            model=ollama_model,
            base_url=ollama_base_url,
            llm_fn=grounding_llm_fn,
        )

        # If unsupported, try one strict regeneration using accepted evidence.
        if grounding_result.status == "Unsupported" and grounding_result.ollama_available:
            regeneration_attempted = True
            feedback = "; ".join(grounding_result.unsupported_claims) or grounding_result.explanation
            regenerated = generate_answer(
                query=cleaned_query,
                evidence=filter_result.accepted,
                unsupported_feedback=feedback,
                model=ollama_model,
                base_url=ollama_base_url,
                llm_fn=generate_llm_fn,
            )
            if not regenerated.abstained and regenerated.ollama_available:
                reverified = verify_grounding(
                    query=cleaned_query,
                    answer=regenerated.answer,
                    evidence=filter_result.accepted,
                    model=ollama_model,
                    base_url=ollama_base_url,
                    llm_fn=grounding_llm_fn,
                )
                if reverified.status == "Unsupported":
                    # Abstain rather than return an answer with ungrounded claims.
                    generation_result.answer = build_abstention_message(
                        cleaned_query,
                        reason="answer verification detected unsupported claims even after regeneration",
                    )
                    generation_result.abstained = True
                    grounding_result = reverified
                    final_grounding_status = "Abstained"
                else:
                    generation_result = regenerated
                    grounding_result = reverified
                    final_grounding_status = reverified.status
            else:
                generation_result.answer = build_abstention_message(
                    cleaned_query,
                    reason="unable to regenerate a verified grounded answer",
                )
                generation_result.abstained = True
                final_grounding_status = "Abstained"
        else:
            final_grounding_status = grounding_result.status

        s8_status = "Failed" if not grounding_result.ollama_available else "Completed"
        tracer.record_stage(
            8,
            "Grounding Verification",
            s8_status,
            _elapsed_ms(s8),
            grounding_result.explanation,
            {
                "status": final_grounding_status,
                "verdict": grounding_result.status,
                "unsupported_claims": grounding_result.unsupported_claims,
                "regeneration_attempted": regeneration_attempted,
                "abstained_after_check": generation_result.abstained,
            },
        )

    all_graded = list(corrective_result.combined_graded_chunks) + list(web_outcome.graded_results)
    trace = PipelineTrace(
        query_id=query_id,
        conversation_id=conversation_id,
        mode="crag",
        original_query=query,
        cleaned_query=cleaned_query,
        rewritten_query=corrective_result.final_rewritten_query,
        intent=intent,
        retrieved_chunks=initial_chunks,
        relevance_grades=all_graded,
        internal_outcome=corrective_result.final_outcome,
        web_outcome=web_outcome,
        accepted_evidence=filter_result.accepted,
        rejected_evidence=filter_result.rejected,
        answer=generation_result.answer,
        abstained=generation_result.abstained,
        citations=generation_result.citations,
        grounding_status=final_grounding_status,
        grounding_result=grounding_result,
        regeneration_attempted=regeneration_attempted,
        stages=tracer.stages,
        stage_timings_ms=tracer.stage_timings_ms,
        total_latency_ms=round(_elapsed_ms(pipeline_start), 2),
        created_at=utc_now_iso(),
    )

    if persist_trace_record:
        persist_trace(trace, db_path=db_path)
    return trace


def run_basic_rag_pipeline(
    query: str,
    conversation_id: str | None = None,
    category: str | None = None,
    document_ids: list[str] | None = None,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
    persist_trace_record: bool = True,
    db_path: Path | None = None,
    index_dir: Path | None = None,
    embed_fn: EmbedFn | None = None,
    generate_llm_fn: LLMCallable | None = None,
) -> PipelineTrace:
    """
    Execute standard Basic RAG (Retrieve top-k -> Generate directly) for comparison against CRAG.
    """
    pipeline_start = time.perf_counter()
    runtime_cfg = _resolve_runtime_config(db_path=db_path)

    effective_top_k = top_k if top_k is not None else runtime_cfg["retrieval_top_k"]
    effective_similarity = (
        similarity_threshold
        if similarity_threshold is not None
        else runtime_cfg["similarity_threshold"]
    )
    ollama_model = runtime_cfg["ollama_model"]
    ollama_base_url = runtime_cfg["ollama_base_url"]
    embedding_model = runtime_cfg["embedding_model"]

    query_id = str(uuid.uuid4())
    tracer = TraceBuilder(
        query_id=query_id,
        original_query=query,
        mode="basic_rag",
        conversation_id=conversation_id,
    )

    s1 = time.perf_counter()
    cleaned_query = preprocess_query(query)
    intent = detect_query_intent(cleaned_query)
    tracer.record_stage(
        1,
        "Query Analysis",
        "Completed" if cleaned_query else "Failed",
        _elapsed_ms(s1),
        f"Preprocessed query with intent '{intent}' (Basic RAG mode).",
        {"cleaned_query": cleaned_query, "intent": intent},
    )

    s2 = time.perf_counter()
    retrieved = retrieve_chunks(
        query=cleaned_query,
        top_k=effective_top_k,
        category=category,
        document_ids=document_ids,
        min_similarity=effective_similarity,
        embedding_model=embedding_model,
        db_path=db_path,
        index_dir=index_dir,
        embed_fn=embed_fn,
    )
    tracer.record_stage(
        2,
        "Document Retrieval",
        "Completed",
        _elapsed_ms(s2),
        f"Retrieved {len(retrieved)} chunk(s) directly for Basic RAG.",
        {"retrieved_count": len(retrieved)},
    )

    for num, name, note in [
        (3, "Relevance Evaluation", "Skipped in Basic RAG mode."),
        (4, "Query Correction", "Skipped in Basic RAG mode."),
        (5, "External Web Retrieval", "Skipped in Basic RAG mode."),
        (6, "Evidence Filtering", "Basic RAG passes all retrieved chunks directly without filtering."),
    ]:
        tracer.record_stage(num, name, "Skipped", 0.0, note)

    s7 = time.perf_counter()
    generation_result = generate_answer(
        query=cleaned_query,
        evidence=retrieved,
        model=ollama_model,
        base_url=ollama_base_url,
        llm_fn=generate_llm_fn,
    )
    tracer.record_stage(
        7,
        "Answer Generation",
        "Failed" if not generation_result.ollama_available else "Completed",
        _elapsed_ms(s7),
        (
            "Generated answer directly from ungraded top-k chunks."
            if not generation_result.abstained
            else "No chunks retrieved; abstained."
        ),
        {"citations_count": len(generation_result.citations), "abstained": generation_result.abstained},
    )

    tracer.record_stage(8, "Grounding Verification", "Skipped", 0.0, "Skipped in Basic RAG mode.")

    # Convert raw retrieved chunks into ungraded placeholder objects for trace parity.
    unfiltered_as_accepted = [
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
            relevance="relevant",
            confidence=max(0.0, min(1.0, chunk.similarity_score)),
            reason="Passed directly without LLM grading (Basic RAG mode).",
        )
        for chunk in retrieved
    ]

    trace = PipelineTrace(
        query_id=query_id,
        conversation_id=conversation_id,
        mode="basic_rag",
        original_query=query,
        cleaned_query=cleaned_query,
        intent=intent,
        retrieved_chunks=retrieved,
        accepted_evidence=unfiltered_as_accepted,
        answer=generation_result.answer,
        abstained=generation_result.abstained,
        citations=generation_result.citations,
        grounding_status="Skipped",
        stages=tracer.stages,
        stage_timings_ms=tracer.stage_timings_ms,
        total_latency_ms=round(_elapsed_ms(pipeline_start), 2),
        created_at=utc_now_iso(),
    )

    if persist_trace_record:
        persist_trace(trace, db_path=db_path)
    return trace

