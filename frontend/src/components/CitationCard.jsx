import { useState } from 'react';
import { ChevronDown, ChevronRight, ExternalLink, FileText, Globe } from 'lucide-react';
import { formatPercent } from '../utils/formatters';

export function CitationCard({ citation }) {
  const [expanded, setExpanded] = useState(false);
  const isWeb = citation.source_type === 'web';

  return (
    <div className="rounded-xl border border-slate-800/90 bg-[#0B1020]/70 transition hover:border-slate-700">
      <button
        type="button"
        onClick={() => setExpanded((prev) => !prev)}
        className="flex w-full items-center justify-between gap-3 px-3.5 py-2.5 text-left"
      >
        <div className="flex min-w-0 items-center gap-2.5">
          <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-lg bg-[#22D3C5]/15 text-xs font-semibold text-[#22D3C5]">
            {citation.source_number}
          </span>
          {isWeb ? (
            <Globe className="h-4 w-4 shrink-0 text-[#A78BFA]" />
          ) : (
            <FileText className="h-4 w-4 shrink-0 text-[#22D3C5]" />
          )}
          <div className="min-w-0">
            <div className="truncate text-xs font-medium text-slate-200">
              {citation.document_name}
            </div>
            <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-400">
              <span>{citation.category}</span>
              {citation.page_number !== null && citation.page_number !== undefined && (
                <span>• Page {citation.page_number}</span>
              )}
              <span>• Sim {Number(citation.similarity_score || 0).toFixed(2)}</span>
              {citation.confidence !== null && citation.confidence !== undefined && (
                <span>• Conf {formatPercent(citation.confidence)}</span>
              )}
            </div>
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-2">
          {isWeb && citation.url && (
            <a
              href={citation.url}
              target="_blank"
              rel="noreferrer"
              onClick={(e) => e.stopPropagation()}
              className="rounded-md p-1 text-slate-400 transition hover:bg-slate-800 hover:text-[#22D3C5]"
              title="Open external source"
            >
              <ExternalLink className="h-3.5 w-3.5" />
            </a>
          )}
          {expanded ? (
            <ChevronDown className="h-4 w-4 text-slate-400" />
          ) : (
            <ChevronRight className="h-4 w-4 text-slate-400" />
          )}
        </div>
      </button>

      {expanded && (
        <div className="border-t border-slate-800/80 px-3.5 py-2.5 text-xs leading-relaxed text-slate-300">
          <p className="whitespace-pre-wrap">{citation.snippet}</p>
          {citation.url && (
            <a
              href={citation.url}
              target="_blank"
              rel="noreferrer"
              className="mt-2 inline-flex items-center gap-1 text-[#22D3C5] hover:underline"
            >
              <span>{citation.url}</span>
              <ExternalLink className="h-3 w-3" />
            </a>
          )}
        </div>
      )}
    </div>
  );
}

