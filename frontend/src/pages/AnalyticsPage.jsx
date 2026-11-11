import { useEffect, useState } from 'react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  Activity,
  BarChart3,
  Clock,
  Database,
  Globe,
  Loader2,
  Play,
  RefreshCw,
  ShieldCheck,
  Sparkles,
} from 'lucide-react';
import { EmptyState } from '../components/EmptyState';
import { api } from '../services/api';
import { formatLatency } from '../utils/formatters';

const GROUNDING_COLORS = ['#10B981', '#F59E0B', '#F43F5E', '#A78BFA', '#64748B'];
const OUTCOME_COLORS = {
  CORRECT: '#10B981',
  AMBIGUOUS: '#F59E0B',
  INCORRECT: '#F43F5E',
};

function MetricCard({ title, value, subtitle, icon: Icon, accent = 'text-[#22D3C5]' }) {
  return (
    <div className="rounded-2xl border border-slate-800/90 bg-[#131B2F] p-4">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-slate-400">{title}</span>
        {Icon && <Icon className={`h-4 w-4 ${accent}`} />}
      </div>
      <div className="mt-2 text-2xl font-bold text-white">{value}</div>
      {subtitle && <div className="mt-1 text-xs text-slate-400">{subtitle}</div>}
    </div>
  );
}

export function AnalyticsPage() {
  const [overview, setOverview] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(true);
  const [runningBench, setRunningBench] = useState(false);
  const [error, setError] = useState(null);

  const loadAnalytics = async () => {
    try {
      setLoading(true);
      setError(null);
      const [ov, cmp] = await Promise.all([
        api.getAnalyticsOverview(),
        api.getComparisonReport(),
      ]);
      setOverview(ov);
      setComparison(cmp);
    } catch (err) {
      setError(err.message || 'Failed to load analytics.');
    } finally {
      setLoading(false);
    }
  };

  const handleRunBenchmark = async () => {
    try {
      setRunningBench(true);
      setError(null);
      const res = await api.runBenchmark({
        top_k: 3,
        web_search_enabled: false,
        use_llm_judge: true,
      });
      setComparison(res);
    } catch (err) {
      setError(err.message || 'Failed to run benchmark.');
    } finally {
      setRunningBench(false);
    }
  };

  useEffect(() => {
    loadAnalytics();
  }, []);

  if (loading && !overview) {
    return (
      <div className="flex h-full items-center justify-center p-8 text-sm text-slate-400">
        Loading real-time analytics from SQLite traces...
      </div>
    );
  }

  const benchmarkSummary = comparison?.available ? comparison.run?.summary : null;
  const comparisonChartData = benchmarkSummary
    ? [
        {
          metric: 'Precision@K',
          BasicRAG: Number(((benchmarkSummary.basic_rag?.retrieval_precision_at_k || 0) * 100).toFixed(1)),
          CRAG: Number(((benchmarkSummary.crag?.retrieval_precision_at_k || 0) * 100).toFixed(1)),
        },
        {
          metric: 'Context Rel.',
          BasicRAG: Number(((benchmarkSummary.basic_rag?.context_relevance || 0) * 100).toFixed(1)),
          CRAG: Number(((benchmarkSummary.crag?.context_relevance || 0) * 100).toFixed(1)),
        },
        {
          metric: 'Faithfulness',
          BasicRAG: Number(((benchmarkSummary.basic_rag?.answer_faithfulness || 0) * 100).toFixed(1)),
          CRAG: Number(((benchmarkSummary.crag?.answer_faithfulness || 0) * 100).toFixed(1)),
        },
        {
          metric: 'Correctness',
          BasicRAG: Number(((benchmarkSummary.basic_rag?.answer_correctness || 0) * 100).toFixed(1)),
          CRAG: Number(((benchmarkSummary.crag?.answer_correctness || 0) * 100).toFixed(1)),
        },
      ]
    : [];

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-4 sm:p-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-white">Analytics & Evaluation</h1>
          <p className="mt-0.5 text-sm text-slate-400">
            Observed telemetry computed exclusively from persisted queries and benchmark runs
          </p>
        </div>
        <button
          type="button"
          onClick={loadAnalytics}
          className="inline-flex items-center gap-1.5 rounded-xl border border-slate-800 bg-[#131B2F] px-3 py-2 text-xs font-medium text-slate-200 transition hover:border-[#22D3C5]/40 hover:text-[#22D3C5]"
        >
          <RefreshCw className="h-3.5 w-3.5" />
          <span>Refresh</span>
        </button>
      </div>

      {error && (
        <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          {error}
        </div>
      )}

      {/* Top KPI Grid */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Total Queries"
          value={overview?.total_queries ?? 0}
          subtitle={`${overview?.crag_queries ?? 0} CRAG • ${overview?.basic_rag_queries ?? 0} Basic RAG`}
          icon={Activity}
        />
        <MetricCard
          title="Average Latency"
          value={overview?.has_data ? formatLatency(overview.avg_latency_ms) : '—'}
          subtitle="End-to-end execution time"
          icon={Clock}
          accent="text-[#A78BFA]"
        />
        <MetricCard
          title="Internal Success Rate"
          value={overview?.has_data ? `${overview.internal_success_rate}%` : '—'}
          subtitle={`Correction freq: ${overview?.correction_frequency ?? 0}%`}
          icon={Sparkles}
          accent="text-emerald-400"
        />
        <MetricCard
          title="Web Fallback & Relevance"
          value={overview?.has_data ? `${overview.web_fallback_frequency}%` : '—'}
          subtitle={`Avg relevance confidence: ${overview?.avg_retrieval_relevance ?? 0}`}
          icon={Globe}
        />
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <MetricCard
          title="Indexed Knowledge Base"
          value={`${overview?.document_count ?? 0} docs`}
          subtitle={`${overview?.total_chunks ?? 0} searchable vector chunks`}
          icon={Database}
        />
        <MetricCard
          title="Grounding Verification"
          value={
            overview?.grounding_breakdown?.length
              ? `${overview.grounding_breakdown.find((g) => g.name === 'Supported')?.value || 0} Supported`
              : 'No checks yet'
          }
          subtitle="Verified across CRAG responses"
          icon={ShieldCheck}
          accent="text-emerald-400"
        />
      </div>

      {/* Main Charts or Empty State */}
      {!overview?.has_data ? (
        <EmptyState
          icon={BarChart3}
          title="No Query Telemetry Yet"
          description="Send queries in the Chat Assistant to record real latencies, internal grading outcomes, query corrections, and grounding verification statistics."
        />
      ) : (
        <div className="grid gap-6 lg:grid-cols-2">
          {/* Queries Over Time */}
          <div className="rounded-2xl border border-slate-800/90 bg-[#131B2F] p-5">
            <h3 className="mb-4 text-sm font-semibold text-white">Queries Over Time</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={overview.queries_over_time}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                  <XAxis dataKey="date" stroke="#94A3B8" fontSize={12} />
                  <YAxis stroke="#94A3B8" fontSize={12} allowDecimals={false} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0B1020',
                      borderColor: '#334155',
                      borderRadius: '0.75rem',
                      color: '#F8FAFC',
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="queries"
                    stroke="#22D3C5"
                    fill="#22D3C5"
                    fillOpacity={0.2}
                    name="Queries"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Retrieval Outcome Distribution */}
          <div className="rounded-2xl border border-slate-800/90 bg-[#131B2F] p-5">
            <h3 className="mb-4 text-sm font-semibold text-white">
              Internal Retrieval Outcome Distribution
            </h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={overview.correction_distribution}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                  <XAxis dataKey="name" stroke="#94A3B8" fontSize={12} />
                  <YAxis stroke="#94A3B8" fontSize={12} allowDecimals={false} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0B1020',
                      borderColor: '#334155',
                      borderRadius: '0.75rem',
                      color: '#F8FAFC',
                    }}
                  />
                  <Bar dataKey="value" name="Queries" radius={[6, 6, 0, 0]}>
                    {overview.correction_distribution.map((entry) => (
                      <Cell
                        key={entry.name}
                        fill={OUTCOME_COLORS[entry.name] || '#22D3C5'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Response Latency History */}
          <div className="rounded-2xl border border-slate-800/90 bg-[#131B2F] p-5">
            <h3 className="mb-4 text-sm font-semibold text-white">
              Recent Response Latency (ms)
            </h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={overview.latency_history}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                  <XAxis dataKey="label" stroke="#94A3B8" fontSize={12} />
                  <YAxis stroke="#94A3B8" fontSize={12} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0B1020',
                      borderColor: '#334155',
                      borderRadius: '0.75rem',
                      color: '#F8FAFC',
                    }}
                  />
                  <Bar dataKey="latency_ms" fill="#A78BFA" name="Latency (ms)" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Grounding Verification Breakdown */}
          <div className="rounded-2xl border border-slate-800/90 bg-[#131B2F] p-5">
            <h3 className="mb-4 text-sm font-semibold text-white">
              Grounding Verification Results
            </h3>
            {overview.grounding_breakdown.length === 0 ? (
              <div className="flex h-64 items-center justify-center text-xs text-slate-400">
                No CRAG grounding checks recorded yet.
              </div>
            ) : (
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={overview.grounding_breakdown}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      label
                    >
                      {overview.grounding_breakdown.map((entry, idx) => (
                        <Cell
                          key={entry.name}
                          fill={GROUNDING_COLORS[idx % GROUNDING_COLORS.length]}
                        />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#0B1020',
                        borderColor: '#334155',
                        borderRadius: '0.75rem',
                        color: '#F8FAFC',
                      }}
                    />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Basic RAG vs CRAG Benchmark Section */}
      <div className="rounded-2xl border border-slate-800/90 bg-[#131B2F] p-5">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
          <div>
            <h3 className="text-base font-semibold text-white">
              Basic RAG vs Adaptive CRAG Benchmark
            </h3>
            <p className="text-xs text-slate-400">
              {comparison?.available
                ? `Run ID: ${comparison.run.id} • Dataset: ${comparison.run.dataset_name}`
                : 'Generated strictly from actual evaluation runs'}
            </p>
          </div>
          <button
            type="button"
            disabled={runningBench}
            onClick={handleRunBenchmark}
            className="inline-flex items-center gap-2 rounded-xl bg-[#22D3C5] px-3.5 py-2 text-xs font-semibold text-[#0B1020] transition hover:bg-[#1ec2b5] disabled:opacity-50"
          >
            {runningBench ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Play className="h-3.5 w-3.5" />
            )}
            <span>{runningBench ? 'Running Benchmark...' : 'Run Benchmark'}</span>
          </button>
        </div>

        {!comparison?.available ? (
          <EmptyState
            icon={BarChart3}
            title="No Benchmark Run Recorded Yet"
            description="Run the evaluation benchmark script against the labelled test dataset to populate real Basic RAG vs CRAG precision, faithfulness, correctness, and latency comparisons."
          />
        ) : (
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={comparisonChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                  <XAxis dataKey="metric" stroke="#94A3B8" fontSize={12} />
                  <YAxis stroke="#94A3B8" fontSize={12} domain={[0, 100]} unit="%" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0B1020',
                      borderColor: '#334155',
                      borderRadius: '0.75rem',
                      color: '#F8FAFC',
                    }}
                  />
                  <Legend />
                  <Bar dataKey="BasicRAG" fill="#A78BFA" name="Basic RAG (%)" radius={[6, 6, 0, 0]} />
                  <Bar dataKey="CRAG" fill="#22D3C5" name="Adaptive CRAG (%)" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="border-b border-slate-800 text-slate-400">
                  <tr>
                    <th className="py-2 pr-4">Metric</th>
                    <th className="py-2 pr-4">Basic RAG</th>
                    <th className="py-2">CRAG</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/70 text-slate-200">
                  <tr>
                    <td className="py-2.5 pr-4">Retrieval Precision@K</td>
                    <td className="py-2.5 pr-4 font-mono">
                      {((benchmarkSummary.basic_rag?.retrieval_precision_at_k || 0) * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 font-mono text-[#22D3C5]">
                      {((benchmarkSummary.crag?.retrieval_precision_at_k || 0) * 100).toFixed(1)}%
                    </td>
                  </tr>
                  <tr>
                    <td className="py-2.5 pr-4">Retrieval Recall@K</td>
                    <td className="py-2.5 pr-4 font-mono">
                      {((benchmarkSummary.basic_rag?.retrieval_recall_at_k || 0) * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 font-mono text-[#22D3C5]">
                      {((benchmarkSummary.crag?.retrieval_recall_at_k || 0) * 100).toFixed(1)}%
                    </td>
                  </tr>
                  <tr>
                    <td className="py-2.5 pr-4">Answer Faithfulness</td>
                    <td className="py-2.5 pr-4 font-mono">
                      {((benchmarkSummary.basic_rag?.answer_faithfulness || 0) * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 font-mono text-[#22D3C5]">
                      {((benchmarkSummary.crag?.answer_faithfulness || 0) * 100).toFixed(1)}%
                    </td>
                  </tr>
                  <tr>
                    <td className="py-2.5 pr-4">Answer Correctness</td>
                    <td className="py-2.5 pr-4 font-mono">
                      {((benchmarkSummary.basic_rag?.answer_correctness || 0) * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 font-mono text-[#22D3C5]">
                      {((benchmarkSummary.crag?.answer_correctness || 0) * 100).toFixed(1)}%
                    </td>
                  </tr>
                  <tr>
                    <td className="py-2.5 pr-4">Avg Latency</td>
                    <td className="py-2.5 pr-4 font-mono">
                      {formatLatency(benchmarkSummary.basic_rag?.avg_latency_ms)}
                    </td>
                    <td className="py-2.5 font-mono">
                      {formatLatency(benchmarkSummary.crag?.avg_latency_ms)}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

