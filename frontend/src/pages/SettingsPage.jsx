import { useEffect, useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  Cpu,
  KeyRound,
  RefreshCw,
  Save,
  Sliders,
} from 'lucide-react';
import { api } from '../services/api';

export function SettingsPage({ health, onSettingsSaved }) {
  const [form, setForm] = useState({
    ollama_model: 'llama3.2:3b',
    ollama_base_url: 'http://127.0.0.1:11434',
    embedding_model: 'sentence-transformers/all-MiniLM-L6-v2',
    web_search_enabled: true,
    retrieval_top_k: 5,
    similarity_threshold: 0.25,
    max_correction_attempts: 1,
    chunk_size: 500,
    chunk_overlap: 100,
  });
  const [tavilyConfigured, setTavilyConfigured] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [savedNotice, setSavedNotice] = useState(null);

  const loadSettings = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.getSettings();
      setTavilyConfigured(Boolean(data.tavily_configured));
      setForm({
        ollama_model: data.ollama_model,
        ollama_base_url: data.ollama_base_url,
        embedding_model: data.embedding_model,
        web_search_enabled: Boolean(data.web_search_enabled),
        retrieval_top_k: Number(data.retrieval_top_k),
        similarity_threshold: Number(data.similarity_threshold),
        max_correction_attempts: Number(data.max_correction_attempts),
        chunk_size: Number(data.chunk_size),
        chunk_overlap: Number(data.chunk_overlap),
      });
    } catch (err) {
      setError(err.message || 'Failed to load settings.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSettings();
  }, []);

  const handleChange = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
    setSavedNotice(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setSavedNotice(null);

    if (Number(form.chunk_overlap) >= Number(form.chunk_size)) {
      setError('Chunk overlap must be strictly smaller than chunk size.');
      return;
    }

    try {
      setSaving(true);
      const updated = await api.updateSettings({
        ...form,
        retrieval_top_k: Number(form.retrieval_top_k),
        similarity_threshold: Number(form.similarity_threshold),
        max_correction_attempts: Number(form.max_correction_attempts),
        chunk_size: Number(form.chunk_size),
        chunk_overlap: Number(form.chunk_overlap),
      });
      setTavilyConfigured(Boolean(updated.tavily_configured));
      setSavedNotice('Settings saved and applied to active pipeline configuration.');
      onSettingsSaved();
    } catch (err) {
      setError(err.message || 'Failed to update settings.');
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center p-8 text-sm text-slate-400">
        Loading configuration...
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-4xl space-y-6 p-4 sm:p-6">
      <div>
        <h1 className="text-xl font-bold text-white">System Settings</h1>
        <p className="mt-0.5 text-sm text-slate-400">
          Configure local Ollama models, embedding parameters, retrieval thresholds, and chunking settings
        </p>
      </div>

      {error && (
        <div className="flex items-center gap-2 rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          <AlertCircle className="h-4 w-4 shrink-0 text-rose-400" />
          <span>{error}</span>
        </div>
      )}

      {savedNotice && (
        <div className="flex items-center gap-2 rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-200">
          <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-400" />
          <span>{savedNotice}</span>
        </div>
      )}

      {/* Secret & Diagnostics Cards */}
      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-2xl border border-slate-800/90 bg-[#131B2F] p-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-white">
            <KeyRound className="h-4 w-4 text-[#22D3C5]" />
            <span>Tavily Web Search API Key</span>
          </div>
          <div className="mt-2 flex items-center gap-2">
            <span
              className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
                tavilyConfigured
                  ? 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30'
                  : 'bg-amber-500/15 text-amber-300 border border-amber-500/30'
              }`}
            >
              {tavilyConfigured ? 'Configured in backend .env' : 'Not configured (Internal-only mode)'}
            </span>
          </div>
          <p className="mt-2 text-xs text-slate-400">
            API keys are stored exclusively in backend environment variables (<code className="text-[#22D3C5]">TAVILY_API_KEY</code>) and never exposed to the browser.
          </p>
        </div>

        <div className="rounded-2xl border border-slate-800/90 bg-[#131B2F] p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm font-semibold text-white">
              <Cpu className="h-4 w-4 text-[#A78BFA]" />
              <span>Local Ollama Diagnostics</span>
            </div>
            <button
              type="button"
              onClick={onSettingsSaved}
              className="rounded-lg p-1 text-slate-400 hover:bg-slate-800 hover:text-white"
              title="Re-check health"
            >
              <RefreshCw className="h-3.5 w-3.5" />
            </button>
          </div>
          <div className="mt-2 space-y-1 text-xs text-slate-300">
            <div>
              Status:{' '}
              <strong className={health?.ollama?.reachable ? 'text-emerald-300' : 'text-amber-300'}>
                {health?.ollama?.reachable ? 'Reachable' : 'Unavailable (run ollama serve)'}
              </strong>
            </div>
            <div>
              Target Model Available:{' '}
              <strong className={health?.ollama?.model_available ? 'text-emerald-300' : 'text-amber-300'}>
                {health?.ollama?.model_available ? 'Yes' : 'Not detected locally'}
              </strong>
            </div>
            {health?.ollama?.available_models?.length > 0 && (
              <div className="truncate text-slate-400">
                Pulled: {health.ollama.available_models.join(', ')}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Editable Settings Form */}
      <form
        onSubmit={handleSubmit}
        className="space-y-5 rounded-2xl border border-slate-800/90 bg-[#131B2F] p-6"
      >
        <div className="flex items-center gap-2 border-b border-slate-800 pb-3 text-sm font-semibold text-white">
          <Sliders className="h-4 w-4 text-[#22D3C5]" />
          <span>Model & Pipeline Configuration</span>
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <label className="space-y-1.5 text-xs">
            <span className="font-medium text-slate-300">Ollama Model</span>
            <input
              type="text"
              value={form.ollama_model}
              onChange={(e) => handleChange('ollama_model', e.target.value)}
              className="w-full rounded-xl border border-slate-700 bg-[#0B1020] px-3 py-2 text-sm text-white focus:border-[#22D3C5] focus:outline-none"
            />
          </label>

          <label className="space-y-1.5 text-xs">
            <span className="font-medium text-slate-300">Ollama Base URL</span>
            <input
              type="text"
              value={form.ollama_base_url}
              onChange={(e) => handleChange('ollama_base_url', e.target.value)}
              className="w-full rounded-xl border border-slate-700 bg-[#0B1020] px-3 py-2 text-sm text-white focus:border-[#22D3C5] focus:outline-none"
            />
          </label>

          <label className="space-y-1.5 text-xs sm:col-span-2">
            <span className="font-medium text-slate-300">Embedding Model</span>
            <input
              type="text"
              value={form.embedding_model}
              onChange={(e) => handleChange('embedding_model', e.target.value)}
              className="w-full rounded-xl border border-slate-700 bg-[#0B1020] px-3 py-2 text-sm text-white focus:border-[#22D3C5] focus:outline-none"
            />
          </label>

          <label className="space-y-1.5 text-xs">
            <span className="font-medium text-slate-300">Retrieval Top-K (1–20)</span>
            <input
              type="number"
              min={1}
              max={20}
              value={form.retrieval_top_k}
              onChange={(e) => handleChange('retrieval_top_k', e.target.value)}
              className="w-full rounded-xl border border-slate-700 bg-[#0B1020] px-3 py-2 text-sm text-white focus:border-[#22D3C5] focus:outline-none"
            />
          </label>

          <label className="space-y-1.5 text-xs">
            <span className="font-medium text-slate-300">Similarity Threshold (0.0–1.0)</span>
            <input
              type="number"
              step={0.05}
              min={0}
              max={1}
              value={form.similarity_threshold}
              onChange={(e) => handleChange('similarity_threshold', e.target.value)}
              className="w-full rounded-xl border border-slate-700 bg-[#0B1020] px-3 py-2 text-sm text-white focus:border-[#22D3C5] focus:outline-none"
            />
          </label>

          <label className="space-y-1.5 text-xs">
            <span className="font-medium text-slate-300">Max Correction Attempts (0–5)</span>
            <input
              type="number"
              min={0}
              max={5}
              value={form.max_correction_attempts}
              onChange={(e) => handleChange('max_correction_attempts', e.target.value)}
              className="w-full rounded-xl border border-slate-700 bg-[#0B1020] px-3 py-2 text-sm text-white focus:border-[#22D3C5] focus:outline-none"
            />
          </label>

          <div className="flex items-center justify-between rounded-xl border border-slate-800 bg-[#0B1020] px-3.5 py-2.5 text-xs">
            <div>
              <div className="font-medium text-slate-200">Enable External Web Fallback</div>
              <div className="text-[11px] text-slate-400">Uses Tavily when configured</div>
            </div>
            <input
              type="checkbox"
              checked={form.web_search_enabled}
              onChange={(e) => handleChange('web_search_enabled', e.target.checked)}
              className="h-4 w-4 accent-[#22D3C5]"
            />
          </div>

          <label className="space-y-1.5 text-xs">
            <span className="font-medium text-slate-300">Chunk Size (100–4000 chars)</span>
            <input
              type="number"
              min={100}
              max={4000}
              value={form.chunk_size}
              onChange={(e) => handleChange('chunk_size', e.target.value)}
              className="w-full rounded-xl border border-slate-700 bg-[#0B1020] px-3 py-2 text-sm text-white focus:border-[#22D3C5] focus:outline-none"
            />
          </label>

          <label className="space-y-1.5 text-xs">
            <span className="font-medium text-slate-300">Chunk Overlap (0–1000 chars)</span>
            <input
              type="number"
              min={0}
              max={1000}
              value={form.chunk_overlap}
              onChange={(e) => handleChange('chunk_overlap', e.target.value)}
              className="w-full rounded-xl border border-slate-700 bg-[#0B1020] px-3 py-2 text-sm text-white focus:border-[#22D3C5] focus:outline-none"
            />
          </label>
        </div>

        <div className="flex justify-end border-t border-slate-800 pt-4">
          <button
            type="submit"
            disabled={saving}
            className="inline-flex items-center gap-2 rounded-xl bg-[#22D3C5] px-5 py-2.5 text-sm font-semibold text-[#0B1020] transition hover:bg-[#1ec2b5] disabled:opacity-50"
          >
            <Save className="h-4 w-4" />
            <span>{saving ? 'Saving...' : 'Save Settings'}</span>
          </button>
        </div>
      </form>
    </div>
  );
}

