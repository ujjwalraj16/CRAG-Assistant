import { useEffect, useState } from 'react';
import {
  Activity,
  CheckCircle2,
  ExternalLink,
  Filter,
  Globe,
  RefreshCw,
  Search,
  ShieldAlert,
  Sparkles,
} from 'lucide-react';
import { EmptyState } from '../components/EmptyState';
import { StatusBadge } from '../components/StatusBadge';
import { api } from '../services/api';
import { formatDate, formatLatency, formatPercent } from '../utils/formatters';

export function PipelinePage({ selectedQueryId, onSelectQueryId }) {
  const [recentTraces, setRecentTraces] = useState([]);
  const [trace, setTrace] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('stages');

  const loadTracesAndDetail = async (targetQueryId = selectedQueryId) => {
    try {
      setLoading(true);
      setError(null);
      const list = await api.listPipelineTraces(30);
      setRecentTraces(list);

      const idToLoad = targetQueryId || list[0]?.query_id;
      if (idToLoad) {
        if (idToLoad !== selectedQueryId) {
          onSelectQueryId(idToLoad);
        }
        const detail = await api.getPipelineTrace(idToLoad);
        setTrace(detail);
      } else {
        setTrace(null);
      }
    } catch (err) {
      setError(err.message || 'Failed to load pipeline trace.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadTracesAndDetail(selectedQueryId);
  }, [selectedQueryId]);

  if (loading && !trace) {
    return (
      <div className="flex h-full items-center justify-center p-8 text-sm text-slate-400">
        Loading real backend execution trace...
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-4 sm:p-6">
      {/* Header & Trace Picker */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-white">CRAG Pipeline Monitor</h1>
          <p className="mt-0.5 text-sm text-slate-400">
            Live backend execution trace across retrieval, relevance grading, query correction, web fallback, and grounding verification
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {recentTraces.length > 0 && (
            <select
              value={trace?.query_id || ''}
              onChange={(e) => onSelectQueryId(e.target.value)}
              className="max-w-xs rounded-xl border border-slate-800 bg-[#131B2F] px-3 py-2 text-xs text-slate-200 focus:border-[#22D3C5] focus:outline-none"
            >
              {recentTraces.map((item) => (
                <option key={item.query_id} value={item.query_id}>
                  [{item.mode.toUpperCase()}] {item.original_query.slice(0, 48)}
                </option>
              ))}
            </select>
          )}
          <button
            type="button"
            onClick={() => loadTracesAndDetail(trace?.query_id)}
            className="inline-flex items-center gap-1.5 rounded-xl border border-slate-800 bg-[#131B2F] px-3 py-2 text-xs font-medium text-slate-200 transition hover:border-[#22D3C5]/40 hover:text-[#22D3C5]"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          {error}
        </div>
      )}

      {!trace ? (
        <EmptyState
          icon={Activity}
          title="No Pipeline Traces Recorded Yet"
          description="Send a question in the Chat Assistant to execute the 8-stage CRAG pipeline and inspect real backend retrieval, grading, and verification events here."
        />
      ) : (
        <>
          {/* Trace Overview Card */}
          <div className="rounded-2xl border border-slate-800/90 bg-[#131B2F] p-5">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div className="space-y-1.5">
                <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400">
                  <span className="rounded-md bg-slate-800 px-2 py-0.5 font-semibold uppercase text-[#22D3C5]">
                    {trace.mode === 'basic_rag' ? 'Basic RAG' : 'CRAG'}
                  </span>
                  <span>Intent: <strong className="text-slate-200">{trace.intent}</strong></span>
                  <span>•</span>
                  <span className="font-mono">{trace.query_id.slice(0, 8)}</span>
                  <span>•</span>
                  <span>{formatDate(trace.created_at)}</span>
                </div>
                <h2 className="text-base font-semibold text-white">
                  {trace.original_query}
                </h2>
                {trace.rewritten_query && (
                  <p className="text-xs text-amber-300">
                    Rewritten Query: <span className="font-medium">{trace.rewritten_query}</span>
                  </p>
                )}
              </div>

              <div className="flex flex-wrap items-center gap-2">
                {trace.internal_outcome && (
                  <StatusBadge status={trace.internal_outcome} label={`Internal: ${trace.internal_outcome}`} />
                )}
                {trace.web_outcome && (
                  <StatusBadge status={trace.web_outcome.status} label={`Web: ${trace.web_outcome.status}`} />
                )}
                <StatusBadge status={trace.grounding_status} label={`Grounding: ${trace.grounding_status}`} />
                <span className="rounded-full border border-slate-700 bg-[#0B1020] px-3 py-0.5 font-mono text-xs text-slate-200">
                  {formatLatency(trace.total_latency_ms)}
                </span>
              </div>
            </div>
          </div>

          {/* Sub-navigation Tabs */}
          <div className="flex flex-wrap gap-2 border-b border-slate-800/80 pb-3">
            {[
              { id: 'stages', label: `Execution Stages (${trace.stages.length})`, icon: Activity },
              { id: 'grading', label: `Relevance Grades (${trace.relevance_grades.length})`, icon: Search },
              {
                id: 'evidence',
                label: `Evidence Filter (${trace.accepted_evidence.length} Accepted / ${trace.rejected_evidence.length} Rejected)`,
                icon: Filter,
              },
              { id: 'grounding', label: 'Answer & Grounding Check', icon: ShieldAlert },
            ].map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setActiveTab(tab.id)}
                  className={`inline-flex items-center gap-2 rounded-xl px-3.5 py-2 text-xs font-medium transition ${
                    activeTab === tab.id
                      ? 'bg-[#22D3C5] text-[#0B1020]'
                      : 'border border-slate-800 bg-[#131B2F] text-slate-300 hover:text-white'
                  }`}
                >
                  <Icon className="h-3.5 w-3.5" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>

          {/* Tab 1: 8 Execution Stages */}
          {activeTab === 'stages' && (
            <div className="grid gap-4 md:grid-cols-2">
              {trace.stages.map((stage) => (
                <div
                  key={stage.stage_number}
                  className="flex flex-col justify-between rounded-2xl border border-slate-800/90 bg-[#131B2F] p-4 transition hover:border-slate-700"
                >
                  <div>
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2.5">
                        <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-[#0B1020] font-mono text-xs font-semibold text-[#22D3C5] border border-slate-800">
                          {stage.stage_number}
                        </span>
                        <h3 className="text-sm font-semibold text-white">{stage.name}</h3>
                      </div>
                      <StatusBadge status={stage.status} />
                    </div>

                    <p className="mt-3 text-xs leading-relaxed text-slate-300">
                      {stage.explanation}
                    </p>

                    {stage.details && Object.keys(stage.details).length > 0 && (
                      <div className="mt-3 rounded-xl border border-slate-800/80 bg-[#0B1020]/80 p-3">
                        <div className="grid grid-cols-1 gap-1.5 text-[11px]">
                          {Object.entries(stage.details).map(([key, value]) => (
                            <div key={key} className="flex items-start justify-between gap-3">
                              <span className="font-mono text-slate-400">{key}:</span>
                              <span className="break-all text-right font-mono text-slate-200">
                                {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="mt-3 flex items-center justify-between border-t border-slate-800/70 pt-2.5 text-[11px] text-slate-400">
                    <span>Stage {stage.stage_number} of 8</span>
                    <span className="font-mono text-slate-300">
                      {formatLatency(stage.duration_ms)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Tab 2: Relevance Grades */}
          {activeTab === 'grading' && (
            <div className="space-y-3">
              {trace.relevance_grades.length === 0 ? (
                <EmptyState
                  icon={Search}
                  title="No Chunks Graded"
                  description="Either no documents matched the query or the query ran in Basic RAG mode where LLM relevance grading is skipped."
                />
              ) : (
                trace.relevance_grades.map((item) => (
                  <div
                    key={item.chunk_id}
                    className="rounded-2xl border border-slate-800/90 bg-[#131B2F] p-4"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2 text-xs">
                        {item.source_type === 'web' ? (
                          <Globe className="h-4 w-4 text-[#A78BFA]" />
                        ) : (
                          <Sparkles className="h-4 w-4 text-[#22D3C5]" />
                        )}
                        <span className="font-semibold text-white">{item.document_name}</span>
                        <span className="text-slate-400">• {item.category}</span>
                        {item.page_number !== null && item.page_number !== undefined && (
                          <span className="text-slate-400">• Page {item.page_number}</span>
                        )}
                      </div>
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-mono text-xs text-slate-400">
                          Sim: {Number(item.similarity_score).toFixed(3)}
                        </span>
                        <span className="font-mono text-xs text-slate-200">
                          Conf: {formatPercent(item.confidence)}
                        </span>
                        <StatusBadge status={item.relevance} />
                      </div>
                    </div>
                    <p className="mt-2.5 text-xs text-slate-300">{item.text}</p>
                    <div className="mt-2.5 rounded-lg bg-[#0B1020] px-3 py-2 text-xs text-slate-400">
                      <span className="font-semibold text-slate-200">Grader reason:</span> {item.reason}
                    </div>
                  </div>
                ))
              )}
            </div>
          )}

          {/* Tab 3: Accepted vs Rejected Evidence */}
          {activeTab === 'evidence' && (
            <div className="grid gap-5 lg:grid-cols-2">
              <div className="space-y-3">
                <h3 className="flex items-center gap-2 text-sm font-semibold text-emerald-300">
                  <CheckCircle2 className="h-4 w-4" />
                  <span>Accepted Evidence ({trace.accepted_evidence.length})</span>
                </h3>
                {trace.accepted_evidence.length === 0 ? (
                  <p className="rounded-xl border border-slate-800 bg-[#131B2F] p-4 text-xs text-slate-400">
                    No evidence survived filtering.
                  </p>
                ) : (
                  trace.accepted_evidence.map((chunk) => (
                    <div
                      key={chunk.chunk_id}
                      className="rounded-xl border border-emerald-500/20 bg-[#131B2F] p-3.5 text-xs"
                    >
                      <div className="flex items-center justify-between gap-2 font-medium text-white">
                        <span>{chunk.document_name}</span>
                        <span className="font-mono text-emerald-300">
                          {formatPercent(chunk.confidence)}
                        </span>
                      </div>
                      <p className="mt-1.5 text-slate-300">{chunk.text}</p>
                      {chunk.url && (
                        <a
                          href={chunk.url}
                          target="_blank"
                          rel="noreferrer"
                          className="mt-2 inline-flex items-center gap-1 text-[#22D3C5] hover:underline"
                        >
                          <span>{chunk.url}</span>
                          <ExternalLink className="h-3 w-3" />
                        </a>
                      )}
                    </div>
                  ))
                )}
              </div>

              <div className="space-y-3">
                <h3 className="flex items-center gap-2 text-sm font-semibold text-rose-300">
                  <Filter className="h-4 w-4" />
                  <span>Rejected Evidence ({trace.rejected_evidence.length})</span>
                </h3>
                {trace.rejected_evidence.length === 0 ? (
                  <p className="rounded-xl border border-slate-800 bg-[#131B2F] p-4 text-xs text-slate-400">
                    No chunks were rejected during filtering.
                  </p>
                ) : (
                  trace.rejected_evidence.map((item, idx) => (
                    <div
                      key={`${item.chunk.chunk_id}-${idx}`}
                      className="rounded-xl border border-rose-500/20 bg-[#131B2F] p-3.5 text-xs"
                    >
                      <div className="flex items-center justify-between gap-2 font-medium text-slate-200">
                        <span>{item.chunk.document_name}</span>
                        <StatusBadge status={item.chunk.relevance} />
                      </div>
                      <p className="mt-1.5 text-slate-400">{item.chunk.text}</p>
                      <div className="mt-2 rounded-md bg-rose-500/10 px-2.5 py-1 text-rose-200">
                        Rejected: {item.rejection_reason}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}

          {/* Tab 4: Grounding Audit */}
          {activeTab === 'grounding' && (
            <div className="space-y-4 rounded-2xl border border-slate-800/90 bg-[#131B2F] p-5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="text-sm font-semibold text-white">Grounding Verification Report</h3>
                <StatusBadge status={trace.grounding_status} />
              </div>

              {trace.grounding_result ? (
                <div className="space-y-3 text-xs">
                  <p className="text-slate-300">{trace.grounding_result.explanation}</p>
                  {trace.regeneration_attempted && (
                    <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-amber-200">
                      Initial draft failed grounding verification; one regeneration attempt was executed.
                    </div>
                  )}
                  {trace.grounding_result.unsupported_claims.length > 0 && (
                    <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-3">
                      <div className="font-semibold text-rose-300">Detected Unsupported Claims:</div>
                      <ul className="mt-1.5 list-disc pl-5 text-rose-200">
                        {trace.grounding_result.unsupported_claims.map((claim, idx) => (
                          <li key={idx}>{claim}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-xs text-slate-400">
                  Grounding verification was skipped or the assistant abstained due to insufficient evidence.
                </p>
              )}

              <div className="border-t border-slate-800/80 pt-3">
                <div className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Final Answer Returned
                </div>
                <p className="whitespace-pre-wrap text-sm text-slate-200">{trace.answer}</p>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

