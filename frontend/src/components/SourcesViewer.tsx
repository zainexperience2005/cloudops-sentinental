import React, { useState } from 'react';
import type { SourceItem } from '../types';
import { BookOpen, ExternalLink, ChevronDown, ChevronUp, FileCode } from 'lucide-react';

interface Props {
  sources: SourceItem[];
}

export const SourcesViewer: React.FC<Props> = ({ sources }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!sources || sources.length === 0) {
    return null;
  }

  const internalCount = sources.filter((s) => s.type === 'internal').length;
  const webCount = sources.filter((s) => s.type === 'web').length;

  return (
    <div className="mt-3 rounded-lg border border-slate-800/90 bg-slate-900/40 text-xs">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between px-3 py-2 text-slate-300 hover:text-white transition-colors"
      >
        <div className="flex items-center space-x-2">
          <BookOpen className="w-3.5 h-3.5 text-emerald-400" />
          <span className="font-medium text-[11px]">
            Grounded In {sources.length} Evidence Source{sources.length > 1 ? 's' : ''}
          </span>
          <div className="flex items-center space-x-1 text-[10px]">
            {internalCount > 0 && (
              <span className="px-1.5 py-0.5 rounded bg-emerald-950/70 border border-emerald-800 text-emerald-300">
                {internalCount} Runbook{internalCount > 1 ? 's' : ''}
              </span>
            )}
            {webCount > 0 && (
              <span className="px-1.5 py-0.5 rounded bg-blue-950/70 border border-blue-800 text-blue-300">
                {webCount} Web
              </span>
            )}
          </div>
        </div>
        {isExpanded ? (
          <ChevronUp className="w-3.5 h-3.5 text-slate-400" />
        ) : (
          <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
        )}
      </button>

      {isExpanded && (
        <div className="px-3 pb-3 pt-1 space-y-2 border-t border-slate-800/80">
          {sources.map((source, idx) => (
            <div
              key={idx}
              className="p-2 rounded border border-slate-800 bg-slate-950/60 hover:border-slate-700 transition-all"
            >
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center space-x-1.5 min-w-0">
                  <span
                    className={`text-[9px] font-semibold uppercase px-1 py-0.2 rounded ${
                      source.type === 'internal'
                        ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                        : 'bg-blue-950 text-blue-400 border border-blue-800'
                    }`}
                  >
                    {source.type === 'internal' ? 'Internal SOP' : 'Web Resource'}
                  </span>
                  <span className="font-medium text-slate-200 truncate text-[11px]">
                    {source.title || source.source}
                  </span>
                </div>

                {source.page && (
                  <span className="text-[10px] text-slate-400 bg-slate-800 px-1.5 py-0.5 rounded">
                    Page {source.page}
                  </span>
                )}
              </div>

              <div className="text-[10px] text-slate-400 truncate flex items-center gap-1 font-mono">
                {source.type === 'web' && source.url ? (
                  <a
                    href={source.url}
                    target="_blank"
                    rel="noreferrer"
                    className="text-blue-400 hover:underline flex items-center gap-1"
                  >
                    {source.url}
                    <ExternalLink className="w-2.5 h-2.5" />
                  </a>
                ) : (
                  <span className="text-slate-500 flex items-center gap-1">
                    <FileCode className="w-3 h-3 text-slate-500" />
                    {source.source}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
