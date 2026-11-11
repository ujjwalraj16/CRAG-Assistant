"""Reproducible Basic RAG vs Adaptive CRAG benchmark runner."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from typing import Any
import uuid

from app.config import PROJECT_ROOT
from app.crag.pipeline import run_basic_rag_pipeline, run_crag_pipeline
from app.database.db import init_db
from app.database.models import save_benchmark_run, utc_now_iso
from app.evaluation.metrics import (
    aggregate_mode_metrics,
    compute_answer_correctness,
    compute_answer_faithfulness,
    compute_context_relevance,
    retrieval_precision_at_k,
    retrieval_recall_at_k,
)
from app.ingestion.indexer import EmbedFn, copy_file_to_uploads, ingest_document
from app.ollama_client import LLMCallable, check_ollama_health
from app.retrieval.web_search import WebSearchFn
from app.schemas.pipeline import PipelineTrace


DEFAULT_DATASET_PATH = PROJECT_ROOT / "evaluation" / "test_dataset.json"
DEFAULT_CORPUS_DIR = PROJECT_ROOT / "evaluation" / "sample_docs"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "evaluation" / "benchmark_results"


def _evaluate_trace(
    sample: dict[str, Any],
    trace: PipelineTrace,
    top_k: int,
    use_llm_judge: bool = False,
    judge_llm_fn: LLMCallable | None = None,
) -> dict[str, Any]:
    """Compute per-query retrieval and answer metrics from an actual PipelineTrace."""
    if trace.mode == "crag":
        context_docs = [item.document_name for item in trace.accepted_evidence]
        context_texts = [item.text for item in trace.accepted_evidence]
    else:
        context_docs = [item.document_name for item in trace.retrieved_chunks]
        context_texts = [item.text for item in trace.retrieved_chunks]

    relevant_docs = sample.get("relevant_docs") or []
    expected_facts = sample.get("expected_facts") or []
    ground_truth = sample.get("ground_truth_answer") or ""
    unanswerable = bool(sample.get("unanswerable_internally", False))

    precision = retrieval_precision_at_k(context_docs, relevant_docs, k=top_k)
    recall = retrieval_recall_at_k(context_docs, relevant_docs, k=top_k)
    context_rel = compute_context_relevance(context_docs, relevant_docs)
    faithfulness = compute_answer_faithfulness(
        query=sample["query"],
        answer=trace.answer,
        context_texts=context_texts,
        abstained=trace.abstained,
        use_llm_judge=use_llm_judge,
        llm_fn=judge_llm_fn,
    )
    correctness = compute_answer_correctness(
        query=sample["query"],
        answer=trace.answer,
        ground_truth_answer=ground_truth,
        expected_facts=expected_facts,
        abstained=trace.abstained,
        unanswerable_internally=unanswerable,
        use_llm_judge=use_llm_judge,
        llm_fn=judge_llm_fn,
    )

    correction_attempted = any(
        stage.name == "Query Correction" and stage.status == "Completed"
        for stage in trace.stages
    )
    web_fallback_used = bool(
        trace.web_outcome is not None
        and trace.web_outcome.executed
        and trace.web_outcome.status in {"SUCCESS", "NO_RESULTS"}
    )

    return {
        "sample_id": sample["id"],
        "query": sample["query"],
        "mode": trace.mode,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "context_relevance": context_rel,
        "answer_faithfulness": faithfulness,
        "answer_correctness": correctness,
        "latency_ms": trace.total_latency_ms,
        "internal_outcome": trace.internal_outcome,
        "grounding_status": trace.grounding_status,
        "correction_attempted": correction_attempted,
        "web_fallback_used": web_fallback_used,
        "abstained": trace.abstained,
        "rewritten_query": trace.rewritten_query,
        "context_docs": context_docs,
        "answer": trace.answer,
    }


def _render_markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    basic = summary["basic_rag"]
    crag = summary["crag"]
    lines = [
        f"# Basic RAG vs Adaptive CRAG Benchmark Report",
        "",
        f"- **Run ID**: `{report['run_id']}`",
        f"- **Dataset**: `{report['dataset_name']}` ({basic['total_queries']} queries)",
        f"- **Timestamp (UTC)**: {report['run_at']}",
        f"- **Web Search Enabled**: {report['web_search_enabled']}",
        f"- **LLM-as-a-Judge Enabled**: {report['use_llm_judge']}",
        "",
        "## Aggregate Comparison",
        "",
        "| Metric | Basic RAG | Adaptive CRAG |",
        "| --- | ---: | ---: |",
        f"| Retrieval Precision@K | {basic['retrieval_precision_at_k']:.4f} | {crag['retrieval_precision_at_k']:.4f} |",
        f"| Retrieval Recall@K | {basic['retrieval_recall_at_k']:.4f} | {crag['retrieval_recall_at_k']:.4f} |",
        f"| Context Relevance | {basic['context_relevance']:.4f} | {crag['context_relevance']:.4f} |",
        f"| Answer Faithfulness | {basic['answer_faithfulness']:.4f} | {crag['answer_faithfulness']:.4f} |",
        f"| Answer Correctness | {basic['answer_correctness']:.4f} | {crag['answer_correctness']:.4f} |",
        f"| Average Latency (ms) | {basic['avg_latency_ms']:.2f} | {crag['avg_latency_ms']:.2f} |",
        f"| Correction Frequency | {basic['correction_frequency']:.2%} | {crag['correction_frequency']:.2%} |",
        f"| Web Fallback Frequency | {basic['web_fallback_frequency']:.2%} | {crag['web_fallback_frequency']:.2%} |",
        f"| Abstention Rate | {basic['abstention_rate']:.2%} | {crag['abstention_rate']:.2%} |",
        "",
    ]
    return "\n".join(lines)


def run_benchmark(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    top_k: int = 3,
    web_search_enabled: bool = False,
    use_llm_judge: bool = False,
    persist_to_app_db: bool = True,
    app_db_path: Path | None = None,
    embed_fn: EmbedFn | None = None,
    grade_llm_fn: LLMCallable | None = None,
    rewrite_llm_fn: LLMCallable | None = None,
    generate_llm_fn: LLMCallable | None = None,
    grounding_llm_fn: LLMCallable | None = None,
    judge_llm_fn: LLMCallable | None = None,
    search_fn: WebSearchFn | None = None,
) -> dict[str, Any]:
    """
    Execute Basic RAG and Adaptive CRAG on identical documents, embeddings, and test queries.

    Uses an isolated temporary SQLite/FAISS index populated from corpus_dir so benchmarks are
    reproducible and never polluted by unrelated uploads.
    """
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset_name = dataset.get("dataset_name", dataset_path.stem)
    samples = dataset.get("samples") or []
    if not samples:
        raise ValueError(f"Benchmark dataset '{dataset_path}' contains no samples.")

    corpus_files = sorted(
        p for p in corpus_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".pdf", ".docx", ".txt", ".csv"}
    )
    if not corpus_files:
        raise ValueError(f"Corpus directory '{corpus_dir}' contains no supported documents.")

    run_id = f"bench_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    run_at = utc_now_iso()

    with tempfile.TemporaryDirectory(prefix="crag_bench_") as tmp_root:
        tmp_path = Path(tmp_root)
        bench_db = tmp_path / "bench.db"
        bench_index = tmp_path / "faiss"
        bench_uploads = tmp_path / "uploads"
        init_db(bench_db)

        for src_file in corpus_files:
            copied = copy_file_to_uploads(src_file, upload_dir=bench_uploads)
            ingest_document(
                file_path=copied,
                filename=src_file.name,
                category="Benchmark",
                chunk_size=500,
                chunk_overlap=100,
                db_path=bench_db,
                index_dir=bench_index,
                embed_fn=embed_fn,
            )

        basic_records: list[dict[str, Any]] = []
        crag_records: list[dict[str, Any]] = []

        for sample in samples:
            query = sample["query"]

            basic_trace = run_basic_rag_pipeline(
                query=query,
                top_k=top_k,
                persist_trace_record=False,
                db_path=bench_db,
                index_dir=bench_index,
                embed_fn=embed_fn,
                generate_llm_fn=generate_llm_fn,
            )
            basic_records.append(
                _evaluate_trace(
                    sample=sample,
                    trace=basic_trace,
                    top_k=top_k,
                    use_llm_judge=use_llm_judge,
                    judge_llm_fn=judge_llm_fn,
                )
            )

            crag_trace = run_crag_pipeline(
                query=query,
                top_k=top_k,
                web_search_enabled=web_search_enabled,
                persist_trace_record=False,
                db_path=bench_db,
                index_dir=bench_index,
                embed_fn=embed_fn,
                grade_llm_fn=grade_llm_fn,
                rewrite_llm_fn=rewrite_llm_fn,
                generate_llm_fn=generate_llm_fn,
                grounding_llm_fn=grounding_llm_fn,
                search_fn=search_fn,
            )
            crag_records.append(
                _evaluate_trace(
                    sample=sample,
                    trace=crag_trace,
                    top_k=top_k,
                    use_llm_judge=use_llm_judge,
                    judge_llm_fn=judge_llm_fn,
                )
            )

    summary = {
        "basic_rag": aggregate_mode_metrics(basic_records),
        "crag": aggregate_mode_metrics(crag_records),
    }
    report = {
        "run_id": run_id,
        "run_at": run_at,
        "dataset_name": dataset_name,
        "top_k": top_k,
        "web_search_enabled": web_search_enabled,
        "use_llm_judge": use_llm_judge,
        "corpus_files": [p.name for p in corpus_files],
        "summary": summary,
        "per_query": {
            "basic_rag": basic_records,
            "crag": crag_records,
        },
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"{run_id}.json"
    latest_json_path = results_dir / "latest_comparison.json"
    md_path = results_dir / "latest_comparison.md"

    serialized = json.dumps(report, indent=2)
    json_path.write_text(serialized, encoding="utf-8")
    latest_json_path.write_text(serialized, encoding="utf-8")
    md_path.write_text(_render_markdown_report(report), encoding="utf-8")

    if persist_to_app_db:
        init_db(app_db_path)
        save_benchmark_run(
            run_id=run_id,
            dataset_name=dataset_name,
            summary=summary,
            details=report["per_query"],
            db_path=app_db_path,
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Basic RAG vs Adaptive CRAG Benchmark.")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k documents to retrieve per query.")
    parser.add_argument(
        "--web-search",
        action="store_true",
        help="Enable external Tavily web fallback during CRAG benchmark run.",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Use local Ollama as an LLM judge for faithfulness and answer correctness.",
    )
    parser.add_argument(
        "--allow-offline",
        action="store_true",
        help="Allow running even if Ollama is unreachable (for failure-mode diagnostics).",
    )
    args = parser.parse_args()

    health = check_ollama_health()
    if not health["reachable"] and not args.allow_offline:
        print(
            f"Error: Ollama is not reachable at {health['base_url']}.\n"
            f"Start Ollama first (`ollama serve`) and ensure `{health['configured_model']}` "
            "is pulled (`ollama pull llama3.2:3b`), or pass --allow-offline to test offline behavior."
        )
        raise SystemExit(1)

    report = run_benchmark(
        top_k=args.top_k,
        web_search_enabled=args.web_search,
        use_llm_judge=args.llm_judge,
    )
    print(_render_markdown_report(report))
    print(f"Saved full JSON report to {DEFAULT_RESULTS_DIR / 'latest_comparison.json'}")


if __name__ == "__main__":
    main()

