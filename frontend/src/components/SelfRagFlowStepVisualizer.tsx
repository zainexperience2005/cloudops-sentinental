import React from 'react';
import { 
  CheckCircle2, 
  HelpCircle, 
  Search, 
  Globe, 
  FileText, 
  ShieldCheck, 
  Database,
  Sparkles,
  RefreshCw
} from 'lucide-react';

interface Props {
  trace: string[];
  route?: string;
  supportStatus?: string;
  usefulness?: string;
  isStreaming?: boolean;
}

export const SelfRagFlowStepVisualizer: React.FC<Props> = ({
  trace,
  route,
  supportStatus,
  usefulness,
  isStreaming = false,
}) => {
  // Parse trace items into friendly step cards
  const steps = trace.map((item, index) => {
    let icon = <CheckCircle2 className="w-4 h-4 text-emerald-400" />;
    let badge = 'Step';
    let color = 'border-slate-800 bg-slate-900/60 text-slate-300';

    if (item.toLowerCase().includes('memory')) {
      icon = <Database className="w-4 h-4 text-purple-400" />;
      badge = 'Memory';
      color = 'border-purple-900/50 bg-purple-950/20 text-purple-300';
    } else if (item.toLowerCase().includes('retrieval decision')) {
      icon = <HelpCircle className="w-4 h-4 text-amber-400" />;
      badge = 'Router';
      color = 'border-amber-900/50 bg-amber-950/20 text-amber-300';
    } else if (item.toLowerCase().includes('internal retrieval')) {
      icon = <Search className="w-4 h-4 text-emerald-400" />;
      badge = 'Pinecone';
      color = 'border-emerald-900/50 bg-emerald-950/20 text-emerald-300';
    } else if (item.toLowerCase().includes('relevance')) {
      icon = <FileText className="w-4 h-4 text-cyan-400" />;
      badge = 'Grader';
      color = 'border-cyan-900/50 bg-cyan-950/20 text-cyan-300';
    } else if (item.toLowerCase().includes('rewrote') || item.toLowerCase().includes('prepared internet')) {
      icon = <RefreshCw className="w-4 h-4 text-orange-400" />;
      badge = 'Rewriter';
      color = 'border-orange-900/50 bg-orange-950/20 text-orange-300';
    } else if (item.toLowerCase().includes('internet search') || item.toLowerCase().includes('web')) {
      icon = <Globe className="w-4 h-4 text-blue-400" />;
      badge = 'Web Search';
      color = 'border-blue-900/50 bg-blue-950/20 text-blue-300';
    } else if (item.toLowerCase().includes('support check')) {
      icon = <ShieldCheck className="w-4 h-4 text-green-400" />;
      badge = 'Hallucination Check';
      color = 'border-green-900/50 bg-green-950/20 text-green-300';
    } else if (item.toLowerCase().includes('usefulness')) {
      icon = <Sparkles className="w-4 h-4 text-yellow-400" />;
      badge = 'Usefulness';
      color = 'border-yellow-900/50 bg-yellow-950/20 text-yellow-300';
    } else if (item.toLowerCase().includes('generated') || item.toLowerCase().includes('answer')) {
      icon = <Sparkles className="w-4 h-4 text-indigo-400" />;
      badge = 'Synthesis';
      color = 'border-indigo-900/50 bg-indigo-950/20 text-indigo-300';
    }

    return {
      index: index + 1,
      text: item,
      icon,
      badge,
      color,
    };
  });

  return (
    <div className="mt-3 rounded-lg border border-slate-800 bg-slate-950/70 p-3 text-xs">
      <div className="flex items-center justify-between mb-2 pb-2 border-b border-slate-800/80">
        <div className="flex items-center space-x-2">
          <span className="relative flex h-2 w-2">
            {isStreaming && (
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            )}
            <span className={`relative inline-flex rounded-full h-2 w-2 ${isStreaming ? 'bg-emerald-500' : 'bg-slate-400'}`}></span>
          </span>
          <span className="font-semibold text-slate-200 tracking-wide uppercase text-[10px]">
            Self-RAG Step Execution Pipeline ({steps.length} Steps)
          </span>
        </div>

        {route && (
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
            Route: <strong className="text-emerald-400">{route}</strong>
          </span>
        )}
      </div>

      {/* Step Pills Timeline */}
      <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1">
        {steps.map((s) => (
          <div
            key={s.index}
            className={`flex items-start gap-2.5 p-2 rounded-md border text-[11px] leading-relaxed transition-all ${s.color}`}
          >
            <div className="mt-0.5 flex-shrink-0">{s.icon}</div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-1.5">
                <span className="font-mono text-[9px] px-1 py-0.2 rounded bg-black/40 text-slate-400 uppercase">
                  {s.badge}
                </span>
                <span className="text-slate-200 font-medium">{s.text}</span>
              </div>
            </div>
            <span className="text-[10px] text-slate-500 font-mono">#{s.index}</span>
          </div>
        ))}

        {steps.length === 0 && (
          <div className="text-slate-500 italic py-1">No steps logged for this turn.</div>
        )}
      </div>

      {/* Pipeline Summary Bar */}
      {(supportStatus || usefulness) && (
        <div className="mt-2 pt-2 border-t border-slate-800/80 flex flex-wrap items-center gap-2 text-[10px]">
          {supportStatus && (
            <div className="flex items-center gap-1 text-slate-400">
              <span>Grounding:</span>
              <span
                className={`font-semibold px-1.5 py-0.5 rounded ${
                  supportStatus === 'fully_supported'
                    ? 'bg-emerald-950/80 text-emerald-300 border border-emerald-800/60'
                    : supportStatus === 'partially_supported'
                    ? 'bg-amber-950/80 text-amber-300 border border-amber-800/60'
                    : 'bg-red-950/80 text-red-300 border border-red-800/60'
                }`}
              >
                {supportStatus === 'fully_supported'
                  ? 'Grounded (100%)'
                  : supportStatus === 'partially_supported'
                  ? 'Partially Grounded'
                  : 'Unsupported'}
              </span>
            </div>
          )}

          {usefulness && (
            <div className="flex items-center gap-1 text-slate-400">
              <span>Critique:</span>
              <span
                className={`font-semibold px-1.5 py-0.5 rounded ${
                  usefulness === 'useful'
                    ? 'bg-cyan-950/80 text-cyan-300 border border-cyan-800/60'
                    : 'bg-yellow-950/80 text-yellow-300 border border-yellow-800/60'
                }`}
              >
                {usefulness === 'useful' ? 'Directly Actionable' : 'Refinement Needed'}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
