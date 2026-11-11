"""Lightweight LLM-based grounding verification to detect unsupported answer claims."""

from pydantic import ValidationError

from app.crag.generator import build_citations, format_evidence_context
from app.ollama_client import (
    LLMCallable,
    OllamaUnavailableError,
    call_ollama,
    parse_llm_json,
)
from app.schemas.pipeline import GradedChunk, GroundingResult


GROUNDING_SYSTEM_PROMPT = (
    "You are an answer grounding verifier. Check whether every factual claim in the "
    "generated answer is supported by the supplied evidence snippets. "
    "Respond ONLY with valid JSON matching: "
    '{"status": "Supported" | "Partially Supported" | "Unsupported", '
    '"explanation": "<short explanation>", '
    '"unsupported_claims": ["<unsupported claim 1>", ...]}'
)


def verify_grounding(
    query: str,
    answer: str,
    evidence: list[GradedChunk],
    model: str | None = None,
    base_url: str | None = None,
    llm_fn: LLMCallable | None = None,
) -> GroundingResult:
    """
    Verify whether a generated answer is grounded in accepted evidence.

    Note: An LLM grounding check reduces hallucinations but does not mathematically
    guarantee zero hallucinations.
    """
    if not answer.strip():
        return GroundingResult(
            status="Unsupported",
            explanation="Generated answer is empty.",
            unsupported_claims=["Empty response"],
        )

    if not evidence:
        return GroundingResult(
            status="Unsupported",
            explanation="No verified evidence was supplied to support the generated answer.",
            unsupported_claims=["Answer generated without verified evidence"],
        )

    citations = build_citations(evidence)
    evidence_block = format_evidence_context(citations)
    prompt = (
        f"User Question: {query}\n\n"
        f"Supplied Evidence:\n{evidence_block}\n\n"
        f"Generated Answer:\n\"\"\"\n{answer}\n\"\"\"\n\n"
        "Determine if all claims in the Generated Answer are backed by the Supplied Evidence. "
        "Return JSON only."
    )

    try:
        raw_output = (
            llm_fn(prompt, GROUNDING_SYSTEM_PROMPT, True)
            if llm_fn is not None
            else call_ollama(
                prompt=prompt,
                system_prompt=GROUNDING_SYSTEM_PROMPT,
                expect_json=True,
                model=model,
                base_url=base_url,
                temperature=0.0,
            )
        )
    except OllamaUnavailableError as exc:
        return GroundingResult(
            status="Partially Supported",
            explanation=f"Ollama unavailable during grounding verification: {exc}",
            unsupported_claims=[],
            ollama_available=False,
            parse_fallback=True,
        )

    try:
        parsed = parse_llm_json(raw_output)
        raw_claims = parsed.get("unsupported_claims") or []
        if isinstance(raw_claims, str):
            raw_claims = [raw_claims] if raw_claims.strip() else []
        cleaned_claims = [str(claim).strip() for claim in raw_claims if str(claim).strip()]

        return GroundingResult.model_validate(
            {
                "status": parsed.get("status"),
                "explanation": parsed.get("explanation") or "Verification completed.",
                "unsupported_claims": cleaned_claims,
                "ollama_available": True,
                "parse_fallback": False,
            }
        )
    except (ValueError, ValidationError):
        return GroundingResult(
            status="Partially Supported",
            explanation="Malformed LLM grounding response; verify citations manually.",
            unsupported_claims=[],
            ollama_available=True,
            parse_fallback=True,
        )

