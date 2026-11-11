import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  Activity,
  AlertCircle,
  BookOpen,
  Eraser,
  Globe,
  Loader2,
  Paperclip,
  Send,
  ShieldCheck,
  Sparkles,
  Upload,
  X,
} from 'lucide-react';
import { CitationCard } from '../components/CitationCard';
import { StatusBadge } from '../components/StatusBadge';
import { api } from '../services/api';
import { formatLatency } from '../utils/formatters';

const EXAMPLE_PROMPTS = [
  {
    title: 'Warranty & Policy Lookup',
    prompt: 'What is the battery warranty coverage and return window for AeroBook laptops?',
  },
  {
    title: 'Product Comparison',
    prompt: 'Compare the camera specs and battery life of the flagship models in our catalog.',
  },
  {
    title: 'Technical Troubleshooting',
    prompt: 'How do I fix rapid battery drain or charging issues after a firmware update?',
  },
  {
    title: 'External Knowledge Fallback',
    prompt: 'What are the latest USB4 vs Thunderbolt 4 bandwidth standards?',
  },
];

const QUICK_CATEGORIES = [
  'General',
  'Electronics',
  'Laptops',
  'Smartphones',
  'Home Appliances',
  'Fashion',
  'Banking',
  'Insurance',
  'Healthcare',
  'Company Policies',
];

export function ChatPage({
  conversationId,
  messages,
  onMessagesChange,
  onConversationCreated,
  onInspectPipeline,
  onKnowledgeUpdated,
  onClearActiveChat,
}) {
  const [input, setInput] = useState('');
  const [mode, setMode] = useState('crag');
  const [categoryFilter, setCategoryFilter] = useState('All');
  const [categories, setCategories] = useState(['All']);
  const [useWebSearch, setUseWebSearch] = useState(true);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState(null);

  // Inline Upload Modal state
  const [showUpload, setShowUpload] = useState(false);
  const [uploadCategory, setUploadCategory] = useState('General');
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState(null);

  const bottomRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    api
      .listDocuments()
      .then((res) => {
        setCategories(['All', ...(res.categories || [])]);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, sending]);

  const handleSend = async (overrideQuery = null) => {
    const queryText = (overrideQuery !== null ? overrideQuery : input).trim();
    if (!queryText || sending) return;

    setError(null);
    if (overrideQuery === null) setInput('');

    const optimisticUser = {
      id: `local-user-${Date.now()}`,
      role: 'user',
      content: queryText,
      mode,
      sources: [],
    };
    onMessagesChange((prev) => [...prev, optimisticUser]);
    setSending(true);

    try {
      const response = await api.sendChat({
        query: queryText,
        conversation_id: conversationId || null,
        mode,
        category: categoryFilter === 'All' ? null : categoryFilter,
        use_web_search: mode === 'crag' ? useWebSearch : false,
      });

      onConversationCreated(response.conversation_id);

      const assistantMsg = {
        id: `local-assistant-${Date.now()}`,
        role: 'assistant',
        content: response.answer,
        query_id: response.query_id,
        mode: response.mode,
        sources: response.citations || [],
        meta: {
          intent: response.intent,
          internal_outcome: response.internal_outcome,
          web_status: response.web_status,
          grounding_status: response.grounding_status,
          rewritten_query: response.rewritten_query,
          total_latency_ms: response.total_latency_ms,
          abstained: response.abstained,
        },
      };
      onMessagesChange((prev) => [...prev, assistantMsg]);
    } catch (err) {
      setError(err.message || 'Failed to get response from backend.');
    } finally {
      setSending(false);
    }
  };

  const handleFilePicked = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadMessage(null);
    setError(null);
    try {
      const res = await api.uploadDocument(file, uploadCategory);
      setUploadMessage(res.message);
      const docs = await api.listDocuments();
      setCategories(['All', ...(docs.categories || [])]);
      onKnowledgeUpdated();
    } catch (err) {
      setError(err.message || 'Document upload failed.');
    } finally {
      setUploading(false);
      event.target.value = '';
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex h-full flex-col">
      {/* Top Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 bg-[#0D1426]/90 px-4 py-3 sm:px-6">
        <div className="flex items-center gap-3">
          <div>
            <h1 className="text-base font-semibold text-white sm:text-lg">
              Intelligent Assistant
            </h1>
            <p className="text-xs text-slate-400">
              Domain-agnostic adaptive CRAG over uploaded docs and external web fallback
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Execution Mode Switcher */}
          <div className="flex rounded-xl border border-slate-800 bg-[#0B1020] p-0.5 text-xs">
            <button
              type="button"
              onClick={() => setMode('crag')}
              className={`rounded-lg px-2.5 py-1.5 font-medium transition ${
                mode === 'crag'
                  ? 'bg-[#22D3C5] text-[#0B1020]'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Adaptive CRAG
            </button>
            <button
              type="button"
              onClick={() => setMode('basic_rag')}
              className={`rounded-lg px-2.5 py-1.5 font-medium transition ${
                mode === 'basic_rag'
                  ? 'bg-[#A78BFA] text-[#0B1020]'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Basic RAG
            </button>
          </div>

          {/* Category Filter */}
          <select
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value)}
            className="rounded-xl border border-slate-800 bg-[#0B1020] px-2.5 py-1.5 text-xs text-slate-200 focus:border-[#22D3C5] focus:outline-none"
            title="Filter internal retrieval by document category"
          >
            {categories.map((cat) => (
              <option key={cat} value={cat}>
                Category: {cat}
              </option>
            ))}
          </select>

          {/* Web Fallback Toggle */}
          {mode === 'crag' && (
            <button
              type="button"
              onClick={() => setUseWebSearch((prev) => !prev)}
              className={`inline-flex items-center gap-1.5 rounded-xl border px-2.5 py-1.5 text-xs font-medium transition ${
                useWebSearch
                  ? 'border-[#22D3C5]/40 bg-[#22D3C5]/10 text-[#22D3C5]'
                  : 'border-slate-800 bg-[#0B1020] text-slate-400 hover:text-slate-200'
              }`}
              title="Toggle external Tavily web fallback when internal evidence is insufficient"
            >
              <Globe className="h-3.5 w-3.5" />
              <span>Web Fallback: {useWebSearch ? 'On' : 'Off'}</span>
            </button>
          )}

          {/* Clear Chat */}
          <button
            type="button"
            onClick={onClearActiveChat}
            disabled={messages.length === 0 && !conversationId}
            className="inline-flex items-center gap-1.5 rounded-xl border border-slate-800 bg-[#0B1020] px-2.5 py-1.5 text-xs font-medium text-slate-300 transition hover:border-slate-700 hover:text-white disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Eraser className="h-3.5 w-3.5" />
            <span>Clear Chat</span>
          </button>
        </div>
      </div>

      {/* Inline Upload Panel */}
      {showUpload && (
        <div className="border-b border-slate-800/80 bg-[#131B2F]/90 px-4 py-3 sm:px-6">
          <div className="mx-auto flex max-w-4xl flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <Upload className="h-4 w-4 text-[#22D3C5]" />
              <span className="font-medium text-white">Quick Upload to Knowledge Base:</span>
              <select
                value={uploadCategory}
                onChange={(e) => setUploadCategory(e.target.value)}
                className="rounded-lg border border-slate-700 bg-[#0B1020] px-2.5 py-1 text-xs text-slate-200 focus:border-[#22D3C5] focus:outline-none"
              >
                {QUICK_CATEGORIES.map((cat) => (
                  <option key={cat} value={cat}>
                    {cat}
                  </option>
                ))}
              </select>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.docx,.txt,.csv"
                onChange={handleFilePicked}
                className="hidden"
              />
              <button
                type="button"
                disabled={uploading}
                onClick={() => fileInputRef.current?.click()}
                className="inline-flex items-center gap-1.5 rounded-lg bg-[#22D3C5] px-3 py-1.5 font-semibold text-[#0B1020] transition hover:bg-[#1ec2b5] disabled:opacity-50"
              >
                {uploading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Paperclip className="h-3.5 w-3.5" />}
                <span>{uploading ? 'Indexing...' : 'Choose PDF / DOCX / TXT / CSV'}</span>
              </button>
              {uploadMessage && <span className="text-emerald-300">{uploadMessage}</span>}
            </div>
            <button
              type="button"
              onClick={() => setShowUpload(false)}
              className="rounded-lg p-1 text-slate-400 hover:bg-slate-800 hover:text-white"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto px-4 py-6 sm:px-6">
        <div className="mx-auto max-w-4xl space-y-6">
          {error && (
            <div className="flex items-center gap-2.5 rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
              <AlertCircle className="h-4 w-4 shrink-0 text-rose-400" />
              <span className="flex-1">{error}</span>
              <button type="button" onClick={() => setError(null)} className="text-rose-300 hover:text-white">
                <X className="h-4 w-4" />
              </button>
            </div>
          )}

          {messages.length === 0 ? (
            <div className="py-8">
              <div className="mb-8 text-center">
                <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-[#22D3C5]/20 to-[#8B5CF6]/20 border border-[#22D3C5]/30">
                  <Sparkles className="h-7 w-7 text-[#22D3C5]" />
                </div>
                <h2 className="text-2xl font-bold text-white">
                  Ask NEXUS AI Anything About Your Knowledge Base
                </h2>
                <p className="mx-auto mt-2 max-w-xl text-sm text-slate-400">
                  Unlike standard RAG, CRAG evaluates retrieval relevance with an LLM, corrects poor queries, falls back to external web search when needed, and verifies answer grounding.
                </p>
              </div>

              <div className="grid gap-3 sm:grid-cols-2">
                {EXAMPLE_PROMPTS.map((item) => (
                  <button
                    key={item.title}
                    type="button"
                    onClick={() => handleSend(item.prompt)}
                    className="group rounded-2xl border border-slate-800/90 bg-[#131B2F]/80 p-4 text-left transition hover:border-[#22D3C5]/40 hover:bg-[#131B2F]"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold uppercase tracking-wider text-[#22D3C5]">
                        {item.title}
                      </span>
                      <BookOpen className="h-4 w-4 text-slate-500 transition group-hover:text-[#22D3C5]" />
                    </div>
                    <p className="mt-2 text-sm text-slate-200">{item.prompt}</p>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg) => {
              const isUser = msg.role === 'user';
              return (
                <div
                  key={msg.id}
                  className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[92%] sm:max-w-[85%] rounded-2xl px-4 py-3.5 ${
                      isUser
                        ? 'bg-[#22D3C5]/15 border border-[#22D3C5]/30 text-white'
                        : 'bg-[#131B2F] border border-slate-800/90 text-slate-100 w-full'
                    }`}
                  >
                    {isUser ? (
                      <p className="whitespace-pre-wrap text-sm leading-relaxed">{msg.content}</p>
                    ) : (
                      <div>
                        {/* Assistant Status Header */}
                        <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/80 pb-2.5">
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="rounded-md bg-slate-800 px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-[#22D3C5]">
                              {msg.mode === 'basic_rag' ? 'Basic RAG' : 'CRAG'}
                            </span>
                            {msg.meta?.internal_outcome && (
                              <StatusBadge status={msg.meta.internal_outcome} />
                            )}
                            {msg.meta?.grounding_status && msg.meta.grounding_status !== 'Skipped' && (
                              <StatusBadge
                                status={msg.meta.grounding_status}
                                label={`Grounding: ${msg.meta.grounding_status}`}
                              />
                            )}
                            {msg.meta?.intent && (
                              <span className="rounded-full border border-slate-700/80 px-2 py-0.5 text-[11px] text-slate-300">
                                {msg.meta.intent}
                              </span>
                            )}
                            {msg.meta?.total_latency_ms !== undefined && (
                              <span className="text-[11px] font-mono text-slate-400">
                                {formatLatency(msg.meta.total_latency_ms)}
                              </span>
                            )}
                          </div>

                          {msg.query_id && (
                            <button
                              type="button"
                              onClick={() => onInspectPipeline(msg.query_id)}
                              className="inline-flex items-center gap-1.5 rounded-lg border border-[#A78BFA]/30 bg-[#A78BFA]/10 px-2.5 py-1 text-xs font-medium text-[#A78BFA] transition hover:bg-[#A78BFA]/20"
                            >
                              <Activity className="h-3.5 w-3.5" />
                              <span>Inspect Trace</span>
                            </button>
                          )}
                        </div>

                        {msg.meta?.rewritten_query && (
                          <div className="mb-3 rounded-lg border border-amber-500/20 bg-amber-500/5 px-3 py-1.5 text-xs text-amber-200">
                            <span className="font-semibold">Query corrected to:</span>{' '}
                            <span className="italic">{msg.meta.rewritten_query}</span>
                          </div>
                        )}

                        <div className="markdown-body">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                        </div>

                        {/* Expandable Evidence Citations */}
                        {msg.sources && msg.sources.length > 0 && (
                          <div className="mt-4 border-t border-slate-800/80 pt-3">
                            <div className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-slate-400">
                              <ShieldCheck className="h-3.5 w-3.5 text-[#22D3C5]" />
                              <span>Supporting Evidence ({msg.sources.length})</span>
                            </div>
                            <div className="space-y-2">
                              {msg.sources.map((citation, idx) => (
                                <CitationCard
                                  key={`${msg.id}-src-${citation.source_number || idx}`}
                                  citation={citation}
                                />
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              );
            })
          )}

          {sending && (
            <div className="flex items-center gap-3 rounded-2xl border border-[#22D3C5]/25 bg-[#131B2F] px-4 py-3.5 text-sm text-slate-300">
              <Loader2 className="h-4 w-4 animate-spin text-[#22D3C5]" />
              <span>
                Running {mode === 'basic_rag' ? 'Basic RAG retrieval & generation' : '8-stage CRAG retrieval, relevance grading & grounding check'}...
              </span>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </div>

      {/* Chat Composer */}
      <div className="border-t border-slate-800/80 bg-[#0D1426]/95 px-4 py-3.5 sm:px-6">
        <div className="mx-auto flex max-w-4xl items-end gap-2">
          <button
            type="button"
            onClick={() => setShowUpload((prev) => !prev)}
            className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-slate-800 bg-[#131B2F] text-slate-300 transition hover:border-[#22D3C5]/40 hover:text-[#22D3C5]"
            title="Upload document to Knowledge Base"
          >
            <Paperclip className="h-4 w-4" />
          </button>

          <textarea
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your uploaded documents or products..."
            className="max-h-36 min-h-[44px] flex-1 resize-none rounded-xl border border-slate-800 bg-[#131B2F] px-4 py-2.5 text-sm text-white placeholder-slate-500 focus:border-[#22D3C5] focus:outline-none"
          />

          <button
            type="button"
            onClick={() => handleSend()}
            disabled={!input.trim() || sending}
            className="flex h-11 shrink-0 items-center gap-2 rounded-xl bg-[#22D3C5] px-4 text-sm font-semibold text-[#0B1020] transition hover:bg-[#1ec2b5] disabled:cursor-not-allowed disabled:opacity-40"
          >
            {sending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            <span className="hidden sm:inline">Send</span>
          </button>
        </div>
      </div>
    </div>
  );
}

