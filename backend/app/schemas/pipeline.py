"""Schemas for retrieval, grading, query correction, web fallback, generation, grounding, and traces."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


RelevanceLabel = Literal["relevant", "irrelevant"]
RetrievalOutcome = Literal["CORRECT", "INCORRECT", "AMBIGUOUS"]
WebSearchStatus = Literal["SKIPPED", "UNAVAILABLE", "SUCCESS", "NO_RESULTS", "ERROR"]
GroundingVerdict = Literal["Supported", "Partially Supported", "Unsupported"]
TraceGroundingStatus = Literal["Supported", "Partially Supported", "Unsupported", "Abstained", "Skipped"]
StageStatus = Literal["Pending", "Running", "Completed", "Skipped", "Failed"]
PipelineMode = Literal["crag", "basic_rag"]
QueryIntent = Literal[
    "Product information",
    "Product comparison",
    "Technical support",
    "Warranty and policies",
    "General knowledge",
    "Other",
]


class RetrievedChunk(BaseModel):
    """A retrieved document chunk along with similarity score and metadata."""

    chunk_id: str
    document_id: str
    document_name: str
    category: str
    page_number: int | None = None
    chunk_index: int
    text: str
    similarity_score: float = Field(description="Cosine similarity score from FAISS.")
    faiss_id: int | None = None


class RelevanceGrade(BaseModel):
    """Structured LLM judgment on whether a retrieved chunk helps answer a query."""

    relevance: RelevanceLabel
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)

    @field_validator("relevance", mode="before")
    @classmethod
    def normalize_relevance(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("relevance must be a string ('relevant' or 'irrelevant').")
        normalized = value.strip().lower()
        if normalized in {"relevant", "irrelevant"}:
            return normalized
        raise ValueError(f"Unsupported relevance label: {value!r}")

    @field_validator("reason", mode="before")
    @classmethod
    def clean_reason(cls, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("reason must be a non-empty string.")
        return value.strip()


class GradedChunk(BaseModel):
    """A retrieved internal chunk or web snippet paired with its relevance evaluation."""

    chunk_id: str
    document_id: str
    document_name: str
    category: str
    page_number: int | None = None
    chunk_index: int = 0
    text: str
    similarity_score: float = 0.0
    source_type: Literal["internal", "web"] = "internal"
    url: str | None = None
    title: str | None = None
    relevance: RelevanceLabel
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    parse_fallback: bool = False


class GradingSummary(BaseModel):
    """Aggregate classification across all graded chunks for one retrieval pass."""

    outcome: RetrievalOutcome
    graded_chunks: list[GradedChunk] = Field(default_factory=list)
    relevant_count: int = 0
    irrelevant_count: int = 0
    avg_confidence: float = 0.0
    explanation: str
    ollama_available: bool = True


class QueryRewriteResult(BaseModel):
    """Result of rewriting an ambiguous or failed search query."""

    original_query: str
    rewritten_query: str
    missing_information: str | None = None
    reason: str
    rewrite_applied: bool = True
    ollama_available: bool = True


class CorrectiveRetrievalAttempt(BaseModel):
    """Snapshot of a single corrective retrieval iteration."""

    attempt_number: int
    rewritten_query: str
    missing_information: str | None = None
    rewrite_reason: str
    retrieved_count: int
    grading: GradingSummary


class CorrectiveRetrievalResult(BaseModel):
    """Full outcome of internal retrieval plus any corrective rewriting attempts."""

    original_query: str
    cleaned_query: str
    intent: QueryIntent
    initial_chunks: list[RetrievedChunk] = Field(default_factory=list)
    initial_grading: GradingSummary
    correction_attempted: bool = False
    attempts: list[CorrectiveRetrievalAttempt] = Field(default_factory=list)
    final_rewritten_query: str | None = None
    final_outcome: RetrievalOutcome
    combined_graded_chunks: list[GradedChunk] = Field(default_factory=list)
    ollama_available: bool = True


class WebSearchRawResult(BaseModel):
    """Normalized raw result returned by Tavily web search."""

    title: str
    url: str
    content: str
    score: float = 0.0


class WebSearchOutcome(BaseModel):
    """Structured outcome of external web fallback and relevance grading."""

    status: WebSearchStatus
    executed: bool = False
    query_used: str | None = None
    raw_results_count: int = 0
    graded_results: list[GradedChunk] = Field(default_factory=list)
    relevant_count: int = 0
    explanation: str
    error: str | None = None
    ollama_available: bool = True


class RejectedEvidence(BaseModel):
    """An evidence item excluded during filtering together with the rejection reason."""

    chunk: GradedChunk
    rejection_reason: str


class EvidenceFilterResult(BaseModel):
    """Combined accepted and rejected evidence passed downstream to answer generation."""

    accepted: list[GradedChunk] = Field(default_factory=list)
    rejected: list[RejectedEvidence] = Field(default_factory=list)
    internal_accepted_count: int = 0
    web_accepted_count: int = 0
    rejected_count: int = 0


class SourceCitation(BaseModel):
    """User-facing citation attached to chat answers and pipeline traces."""

    source_number: int
    source_type: Literal["internal", "web"]
    document_name: str
    category: str
    page_number: int | None = None
    url: str | None = None
    similarity_score: float = 0.0
    confidence: float | None = None
    snippet: str


class GenerationResult(BaseModel):
    """Output from the evidence-grounded answer generator."""

    answer: str
    abstained: bool = False
    citations: list[SourceCitation] = Field(default_factory=list)
    ollama_available: bool = True
    regenerated: bool = False


class GroundingResult(BaseModel):
    """Structured LLM verification checking if claims are backed by evidence."""

    status: GroundingVerdict
    explanation: str
    unsupported_claims: list[str] = Field(default_factory=list)
    ollama_available: bool = True
    parse_fallback: bool = False

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("status must be a string.")
        normalized = " ".join(value.strip().lower().split())
        mapping = {
            "supported": "Supported",
            "partially supported": "Partially Supported",
            "partially_supported": "Partially Supported",
            "partial": "Partially Supported",
            "unsupported": "Unsupported",
        }
        if normalized in mapping:
            return mapping[normalized]
        raise ValueError(f"Unsupported grounding verdict: {value!r}")

    @field_validator("explanation", mode="before")
    @classmethod
    def clean_explanation(cls, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("explanation must be a non-empty string.")
        return value.strip()


class PipelineStageTrace(BaseModel):
    """Execution record for a single stage in the 8-stage CRAG Pipeline Monitor."""

    stage_number: int
    name: str
    status: StageStatus
    duration_ms: float = 0.0
    explanation: str
    details: dict[str, Any] = Field(default_factory=dict)


class PipelineTrace(BaseModel):
    """Complete execution trace persisted for every CRAG or Basic RAG query."""

    query_id: str
    conversation_id: str | None = None
    mode: PipelineMode = "crag"
    original_query: str
    cleaned_query: str
    rewritten_query: str | None = None
    intent: QueryIntent = "Other"
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    relevance_grades: list[GradedChunk] = Field(default_factory=list)
    internal_outcome: RetrievalOutcome | None = None
    web_outcome: WebSearchOutcome | None = None
    accepted_evidence: list[GradedChunk] = Field(default_factory=list)
    rejected_evidence: list[RejectedEvidence] = Field(default_factory=list)
    answer: str
    abstained: bool = False
    citations: list[SourceCitation] = Field(default_factory=list)
    grounding_status: TraceGroundingStatus = "Skipped"
    grounding_result: GroundingResult | None = None
    regeneration_attempted: bool = False
    stages: list[PipelineStageTrace] = Field(default_factory=list)
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)
    total_latency_ms: float = 0.0
    created_at: str
