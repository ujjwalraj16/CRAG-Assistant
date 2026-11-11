import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  Loader2,
  MinusCircle,
  XCircle,
} from 'lucide-react';

const STATUS_STYLES = {
  Completed: {
    classes: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30',
    Icon: CheckCircle2,
  },
  CORRECT: {
    classes: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30',
    Icon: CheckCircle2,
  },
  Supported: {
    classes: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30',
    Icon: CheckCircle2,
  },
  relevant: {
    classes: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30',
    Icon: CheckCircle2,
  },
  SUCCESS: {
    classes: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30',
    Icon: CheckCircle2,
  },
  Running: {
    classes: 'bg-[#22D3C5]/10 text-[#22D3C5] border-[#22D3C5]/40',
    Icon: Loader2,
    spin: true,
  },
  AMBIGUOUS: {
    classes: 'bg-amber-500/10 text-amber-300 border-amber-500/30',
    Icon: AlertTriangle,
  },
  'Partially Supported': {
    classes: 'bg-amber-500/10 text-amber-300 border-amber-500/30',
    Icon: AlertTriangle,
  },
  Pending: {
    classes: 'bg-slate-700/30 text-slate-300 border-slate-600/40',
    Icon: Clock,
  },
  Skipped: {
    classes: 'bg-slate-800/70 text-slate-400 border-slate-700/60',
    Icon: MinusCircle,
  },
  SKIPPED: {
    classes: 'bg-slate-800/70 text-slate-400 border-slate-700/60',
    Icon: MinusCircle,
  },
  UNAVAILABLE: {
    classes: 'bg-slate-800/70 text-slate-400 border-slate-700/60',
    Icon: MinusCircle,
  },
  Abstained: {
    classes: 'bg-purple-500/15 text-purple-300 border-purple-500/30',
    Icon: AlertTriangle,
  },
  Failed: {
    classes: 'bg-rose-500/10 text-rose-300 border-rose-500/30',
    Icon: XCircle,
  },
  INCORRECT: {
    classes: 'bg-rose-500/10 text-rose-300 border-rose-500/30',
    Icon: XCircle,
  },
  Unsupported: {
    classes: 'bg-rose-500/10 text-rose-300 border-rose-500/30',
    Icon: XCircle,
  },
  irrelevant: {
    classes: 'bg-rose-500/10 text-rose-300 border-rose-500/30',
    Icon: XCircle,
  },
  ERROR: {
    classes: 'bg-rose-500/10 text-rose-300 border-rose-500/30',
    Icon: XCircle,
  },
};

export function StatusBadge({ status, label }) {
  const style = STATUS_STYLES[status] || STATUS_STYLES.Pending;
  const Icon = style.Icon;
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium ${style.classes}`}
    >
      <Icon className={`h-3.5 w-3.5 ${style.spin ? 'animate-spin' : ''}`} />
      <span>{label || status}</span>
    </span>
  );
}

