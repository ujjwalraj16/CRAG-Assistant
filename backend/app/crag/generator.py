"""Evidence-grounded answer generation using local Ollama inference."""

from app.ollama_client import LLMCallable, OllamaUnavailableError, call_ollama
from app.schemas.pipeline import (
    GenerationResult,
    GradedChunk,
    RetrievedChunk,
    SourceCitation,
)


GENERATOR_SYSTEM_PROMPT = (
    "You are an intelligent, domain-agnostic product information and support assistant. "
    "Answer the user's question using ONLY the supplied evidence blocks.\n\n"
    "Strict grounding rules:\n"
    "1. Base every factual claim strictly on the supplied evidence.\n"
    "2. Never invent product specifications, prices, warranty durations, policies, or dates.\n"
    "3. Cite supporting evidence inline using [Source 1], [Source 2], etc.\n"
    "4. If part of the question is not covered by the supplied evidence, explicitly state "
    "what information is unavailable instead of guessing.\n"
    "5. Keep the answer clear, structured, and concise using Markdown."
)


def build_citations(evidence: list[GradedChunk] | list[RetrievedChunk]) -> list[SourceCitation]:
    """Convert accepted chunks or raw retrieved chunks into numbered user-facing citations."""
    citations: list[SourceCitation] = []
    for idx, item in enumerate(evidence, start=1):
        if isinstance(item, GradedChunk):
            citations.append(
                SourceCitation(
                    source_number=idx,
                    source_type=item.source_type,
                    document_name=item.document_name,
                    category=item.category,
                    page_number=item.page_number,
                    url=item.url,
                    similarity_score=item.similarity_score,
                    confidence=item.confidence,
                    snippet=item.text,
                )
            )
        else:
            citations.append(
                SourceCitation(
                    source_number=idx,
                    source_type="internal",
                    document_name=item.document_name,
                    category=item.category,
                    page_number=item.page_number,
                    url=None,
                    similarity_score=item.similarity_score,
                    confidence=None,
                    snippet=item.text,
                )
            )
    return citations


def format_evidence_context(citations: list[SourceCitation]) -> str:
    """Format numbered evidence blocks for the answer generation prompt."""
    blocks: list[str] = []
    for citation in citations:
        location_parts = [f"Category: {citation.category}"]
        if citation.page_number is not None:
            location_parts.append(f"Page {citation.page_number}")
        if citation.url:
            location_parts.append(f"URL: {citation.url}")
        header = (
            f"[Source {citation.source_number}] {citation.document_name} "
            f"({', '.join(location_parts)})"
        )
        blocks.append(f"{header}\n{citation.snippet}")
    return "\n\n---\n\n".join(blocks)


def build_abstention_message(
    query: str,
    reason: str | None = None,
) -> str:
    """Return a clear, honest abstention when verified evidence is insufficient."""
    detail = f" ({reason})" if reason else ""
    return (
        f"I do not have enough verified evidence to reliably answer: **\"{query}\"**{detail}.\n\n"
        "To get a grounded answer, try uploading a relevant document in the **Knowledge Base** "
        "or enabling external Tavily web search in **Settings**."
    )


def generate_answer(
    query: str,
    evidence: list[GradedChunk] | list[RetrievedChunk],
    unsupported_feedback: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    llm_fn: LLMCallable | None = None,
) -> GenerationResult:
    """Generate a cited response grounded exclusively in supplied evidence."""
    citations = build_citations(evidence)
    if not citations:
        return GenerationResult(
            answer=build_abstention_message(
                query,
                reason="no relevant internal or external evidence passed verification",
            ),
            abstained=True,
            citations=[],
            ollama_available=True,
            regenerated=bool(unsupported_feedback),
        )

    evidence_block = format_evidence_context(citations)
    feedback_section = ""
    if unsupported_feedback:
        feedback_section = (
            "\n\nIMPORTANT CORRECTION REQUIRED:\n"
            "Your previous draft included claims not supported by the evidence:\n"
            f"{unsupported_feedback}\n"
            "Regenerate the answer using ONLY facts explicitly present in the supplied sources. "
            "Do not extrapolate or include unverified claims."
        )

    prompt = (
        f"User Question: {query}\n\n"
        f"Supplied Evidence:\n{evidence_block}\n"
        f"{feedback_section}\n\n"
        "Write an accurate, well-structured answer grounded only in the supplied sources, "
        "citing [Source N] for factual claims."
    )

    try:
        answer_text = (
            llm_fn(prompt, GENERATOR_SYSTEM_PROMPT, False)
            if llm_fn is not None
            else call_ollama(
                prompt=prompt,
                system_prompt=GENERATOR_SYSTEM_PROMPT,
                expect_json=False,
                model=model,
                base_url=base_url,
                temperature=0.1,
            )
        )
    except OllamaUnavailableError as exc:
        return GenerationResult(
            answer=(
                "The local Ollama language model service is currently unavailable, so I could "
                f"not synthesize a natural-language answer ({exc}).\n\n"
                "However, relevant supporting sources were retrieved and are attached below for inspection."
            ),
            abstained=True,
            citations=citations,
            ollama_available=False,
            regenerated=bool(unsupported_feedback),
        )

    return GenerationResult(
        answer=answer_text.strip(),
        abstained=False,
        citations=citations,
        ollama_available=True,
        regenerated=bool(unsupported_feedback),
    )

