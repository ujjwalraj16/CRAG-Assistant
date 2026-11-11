import { useEffect, useRef, useState } from 'react';
import {
  AlertCircle,
  BookOpen,
  CheckCircle2,
  FileText,
  Loader2,
  RefreshCw,
  Search,
  Tag,
  Trash2,
  UploadCloud,
} from 'lucide-react';
import { EmptyState } from '../components/EmptyState';
import { api } from '../services/api';
import { formatBytes, formatDate } from '../utils/formatters';

const PRESET_CATEGORIES = [
  'General',
  'Electronics',
  'Smartphones',
  'Laptops',
  'Home Appliances',
  'Fashion',
  'Banking',
  'Insurance',
  'Healthcare',
  'Company Policies',
];

export function KnowledgeBasePage({ onKnowledgeChanged }) {
  const [documents, setDocuments] = useState([]);
  const [categories, setCategories] = useState([]);
  const [search, setSearch] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('All');

  const [uploadCategory, setUploadCategory] = useState('Electronics');
  const [customCategory, setCustomCategory] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [busyDocId, setBusyDocId] = useState(null);
  const [feedback, setFeedback] = useState(null);
  const [error, setError] = useState(null);

  const fileInputRef = useRef(null);

  const fetchDocuments = async (searchTerm = search, category = selectedCategory) => {
    try {
      setLoading(true);
      const data = await api.listDocuments({ search: searchTerm, category });
      setDocuments(data.documents || []);
      setCategories(data.categories || []);
    } catch (err) {
      setError(err.message || 'Failed to load knowledge base documents.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      fetchDocuments(search, selectedCategory);
    }, 180);
    return () => clearTimeout(timer);
  }, [search, selectedCategory]);

  const effectiveUploadCategory = customCategory.trim() || uploadCategory;

  const handleUploadFiles = async (fileList) => {
    const files = Array.from(fileList || []);
    if (files.length === 0) return;

    setUploading(true);
    setError(null);
    setFeedback(null);

    try {
      const messages = [];
      for (const file of files) {
        const res = await api.uploadDocument(file, effectiveUploadCategory);
        messages.push(res.message);
      }
      setFeedback(messages.join(' '));
      await fetchDocuments(search, selectedCategory);
      onKnowledgeChanged();
    } catch (err) {
      setError(err.message || 'Failed to upload document.');
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (doc) => {
    setBusyDocId(doc.id);
    setError(null);
    setFeedback(null);
    try {
      const res = await api.deleteDocument(doc.id);
      setFeedback(res.message);
      await fetchDocuments(search, selectedCategory);
      onKnowledgeChanged();
    } catch (err) {
      setError(err.message || 'Failed to delete document.');
    } finally {
      setBusyDocId(null);
    }
  };

  const handleReindex = async (doc) => {
    setBusyDocId(doc.id);
    setError(null);
    setFeedback(null);
    try {
      const res = await api.reindexDocument(doc.id);
      setFeedback(res.message);
      await fetchDocuments(search, selectedCategory);
      onKnowledgeChanged();
    } catch (err) {
      setError(err.message || 'Failed to re-index document.');
    } finally {
      setBusyDocId(null);
    }
  };

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-4 sm:p-6">
      <div>
        <h1 className="text-xl font-bold text-white">Knowledge Base</h1>
        <p className="mt-0.5 text-sm text-slate-400">
          Upload domain-agnostic documents (PDF, DOCX, TXT, CSV) with category metadata for FAISS vector retrieval
        </p>
      </div>

      {/* Alerts */}
      {error && (
        <div className="flex items-center gap-2 rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          <AlertCircle className="h-4 w-4 shrink-0 text-rose-400" />
          <span>{error}</span>
        </div>
      )}
      {feedback && (
        <div className="flex items-center gap-2 rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-200">
          <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-400" />
          <span>{feedback}</span>
        </div>
      )}

      {/* Upload & Category Card */}
      <div className="rounded-2xl border border-slate-800/90 bg-[#131B2F] p-5 space-y-4">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 text-xs font-medium text-slate-300">
            <Tag className="h-4 w-4 text-[#22D3C5]" />
            <span>Document Category Metadata:</span>
          </div>
          <select
            value={uploadCategory}
            onChange={(e) => setUploadCategory(e.target.value)}
            className="rounded-xl border border-slate-700 bg-[#0B1020] px-3 py-1.5 text-xs text-white focus:border-[#22D3C5] focus:outline-none"
          >
            {PRESET_CATEGORIES.map((cat) => (
              <option key={cat} value={cat}>
                {cat}
              </option>
            ))}
          </select>
          <input
            type="text"
            value={customCategory}
            onChange={(e) => setCustomCategory(e.target.value)}
            placeholder="Or type custom category (optional)"
            className="w-60 rounded-xl border border-slate-700 bg-[#0B1020] px-3 py-1.5 text-xs text-white placeholder-slate-500 focus:border-[#22D3C5] focus:outline-none"
          />
        </div>

        {/* Drag and Drop Zone */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragActive(false);
            handleUploadFiles(e.dataTransfer.files);
          }}
          onClick={() => !uploading && fileInputRef.current?.click()}
          className={`flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-9 text-center transition ${
            dragActive
              ? 'border-[#22D3C5] bg-[#22D3C5]/10'
              : 'border-slate-700/90 bg-[#0B1020]/50 hover:border-[#22D3C5]/50'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.docx,.txt,.csv"
            onChange={(e) => {
              handleUploadFiles(e.target.files);
              e.target.value = '';
            }}
            className="hidden"
          />
          {uploading ? (
            <>
              <Loader2 className="mb-3 h-8 w-8 animate-spin text-[#22D3C5]" />
              <p className="text-sm font-medium text-white">Extracting, chunking & indexing vectors...</p>
            </>
          ) : (
            <>
              <UploadCloud className="mb-3 h-8 w-8 text-[#22D3C5]" />
              <p className="text-sm font-medium text-white">
                Drag and drop documents here, or click to browse
              </p>
              <p className="mt-1 text-xs text-slate-400">
                Supports PDF, DOCX, TXT, and CSV • Active Category:{' '}
                <span className="font-semibold text-[#22D3C5]">{effectiveUploadCategory}</span>
              </p>
            </>
          )}
        </div>
      </div>

      {/* Search & Filter Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="relative flex-1 min-w-[220px] max-w-md">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search documents by name or category..."
            className="w-full rounded-xl border border-slate-800 bg-[#131B2F] py-2 pl-9 pr-4 text-sm text-white placeholder-slate-500 focus:border-[#22D3C5] focus:outline-none"
          />
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          {['All', ...categories].map((cat) => (
            <button
              key={cat}
              type="button"
              onClick={() => setSelectedCategory(cat)}
              className={`rounded-full px-3 py-1 text-xs font-medium transition ${
                selectedCategory === cat
                  ? 'bg-[#22D3C5] text-[#0B1020]'
                  : 'border border-slate-800 bg-[#131B2F] text-slate-300 hover:text-white'
              }`}
            >
              {cat}
            </button>
          ))}
        </div>
      </div>

      {/* Document Listing */}
      {loading ? (
        <div className="py-10 text-center text-sm text-slate-400">Loading documents...</div>
      ) : documents.length === 0 ? (
        <EmptyState
          icon={BookOpen}
          title="No Documents Indexed"
          description="Upload PDFs, Word documents, text files, or product CSV catalogs above to populate the FAISS vector store."
        />
      ) : (
        <div className="overflow-hidden rounded-2xl border border-slate-800/90 bg-[#131B2F]">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-slate-800 bg-[#0D1426] text-xs uppercase tracking-wider text-slate-400">
                <tr>
                  <th className="px-4 py-3">Document</th>
                  <th className="px-4 py-3">Category</th>
                  <th className="px-4 py-3">Chunks</th>
                  <th className="px-4 py-3">Size</th>
                  <th className="px-4 py-3">Uploaded</th>
                  <th className="px-4 py-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70">
                {documents.map((doc) => {
                  const busy = busyDocId === doc.id;
                  return (
                    <tr key={doc.id} className="transition hover:bg-[#18223C]/50">
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2.5">
                          <FileText className="h-4 w-4 shrink-0 text-[#22D3C5]" />
                          <div>
                            <div className="font-medium text-white">{doc.filename}</div>
                            <div className="text-[11px] uppercase text-slate-400">{doc.file_type}</div>
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <span className="rounded-full border border-[#A78BFA]/30 bg-[#A78BFA]/10 px-2.5 py-0.5 text-xs font-medium text-[#A78BFA]">
                          {doc.category}
                        </span>
                      </td>
                      <td className="px-4 py-3 font-mono text-xs text-slate-200">
                        {doc.chunk_count}
                      </td>
                      <td className="px-4 py-3 text-xs text-slate-400">
                        {formatBytes(doc.file_size)}
                      </td>
                      <td className="px-4 py-3 text-xs text-slate-400">
                        {formatDate(doc.created_at)}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <div className="inline-flex items-center gap-2">
                          <button
                            type="button"
                            disabled={busy}
                            onClick={() => handleReindex(doc)}
                            className="inline-flex items-center gap-1 rounded-lg border border-slate-700 bg-[#0B1020] px-2.5 py-1.5 text-xs text-slate-200 transition hover:border-[#22D3C5]/40 hover:text-[#22D3C5] disabled:opacity-40"
                            title="Re-chunk and re-embed document"
                          >
                            <RefreshCw className={`h-3.5 w-3.5 ${busy ? 'animate-spin' : ''}`} />
                            <span>Re-index</span>
                          </button>
                          <button
                            type="button"
                            disabled={busy}
                            onClick={() => handleDelete(doc)}
                            className="inline-flex items-center gap-1 rounded-lg border border-rose-500/30 bg-rose-500/10 px-2.5 py-1.5 text-xs text-rose-300 transition hover:bg-rose-500/20 disabled:opacity-40"
                            title="Delete document and remove vectors"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                            <span>Delete</span>
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

