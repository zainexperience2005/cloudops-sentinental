import React, { useState, useEffect } from 'react';
import { 
  ShieldCheck, 
  RefreshCw, 
  Search, 
  FileText, 
  Globe, 
  Layers, 
  CheckCircle, 
  ChevronDown,
  ChevronUp,
  Clock
} from 'lucide-react';
import type { AuditRecord } from '../types';
import { fetchAudits } from '../services/api';

export const AuditsView: React.FC = () => {
  const [audits, setAudits] = useState<AuditRecord[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [routeFilter, setRouteFilter] = useState<string>('all');
  const [expandedId, setExpandedId] = useState<number | null>(null);

  const loadAudits = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await fetchAudits(100);
      setAudits(data);
    } catch (err: any) {
      setError(err.message || 'Failed to load audit records');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadAudits();
  }, []);

  // Filter audits
  const filteredAudits = audits.filter((a) => {
    const matchesSearch =
      a.question.toLowerCase().includes(searchQuery.toLowerCase()) ||
      a.answer.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesRoute =
      routeFilter === 'all' ||
      (a.route && a.route.toLowerCase().includes(routeFilter.toLowerCase()));

    return matchesSearch && matchesRoute;
  });

  // Calculate metrics
  const totalAudits = audits.length;
  const fullySupportedCount = audits.filter((a) => a.support_status === 'fully_supported').length;
  const groundingRate = totalAudits > 0 ? Math.round((fullySupportedCount / totalAudits) * 100) : 0;
  const webUsedCount = audits.filter((a) => a.used_web === 1).length;
  const runbookCount = audits.filter((a) => a.route && a.route.toLowerCase().includes('runbook')).length;

  return (
    <div className="flex-1 flex flex-col h-full bg-[#090d16] overflow-hidden">
      {/* Top Header */}
      <header className="h-14 border-b border-slate-800/80 bg-slate-950/70 backdrop-blur-md px-6 flex items-center justify-between z-10">
        <div className="flex items-center space-x-3">
          <div className="p-1.5 rounded-lg bg-emerald-950/80 border border-emerald-800/60 text-emerald-400">
            <ShieldCheck className="w-4 h-4" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-white">Self-RAG Incident Audits</h1>
            <p className="text-[11px] text-slate-400">Persistent evaluation & hallucination verification audit logs</p>
          </div>
        </div>

        <button
          onClick={loadAudits}
          disabled={isLoading}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-all disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </header>

      {/* Main Content Area */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
        {/* KPI Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40 glass-panel">
            <div className="flex items-center justify-between text-slate-400 mb-1">
              <span className="text-xs">Total Queries</span>
              <Layers className="w-4 h-4 text-emerald-400" />
            </div>
            <div className="text-2xl font-bold text-white">{totalAudits}</div>
            <div className="text-[10px] text-slate-500 mt-1">Stored in SQLite audit.db</div>
          </div>

          <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40 glass-panel">
            <div className="flex items-center justify-between text-slate-400 mb-1">
              <span className="text-xs">Grounded Rate</span>
              <CheckCircle className="w-4 h-4 text-emerald-400" />
            </div>
            <div className="text-2xl font-bold text-emerald-400">{groundingRate}%</div>
            <div className="text-[10px] text-slate-500 mt-1">{fullySupportedCount} fully supported</div>
          </div>

          <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40 glass-panel">
            <div className="flex items-center justify-between text-slate-400 mb-1">
              <span className="text-xs">Runbook Hits</span>
              <FileText className="w-4 h-4 text-blue-400" />
            </div>
            <div className="text-2xl font-bold text-white">{runbookCount}</div>
            <div className="text-[10px] text-slate-500 mt-1">Pinecone SOP evidence</div>
          </div>

          <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40 glass-panel">
            <div className="flex items-center justify-between text-slate-400 mb-1">
              <span className="text-xs">Web Escalations</span>
              <Globe className="w-4 h-4 text-purple-400" />
            </div>
            <div className="text-2xl font-bold text-white">{webUsedCount}</div>
            <div className="text-[10px] text-slate-500 mt-1">Tavily internet search</div>
          </div>
        </div>

        {/* Filter Controls */}
        <div className="flex flex-col md:flex-row gap-3 items-center justify-between">
          <div className="relative w-full md:w-80">
            <Search className="w-4 h-4 absolute left-3 top-2.5 text-slate-500" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search audit questions or answers..."
              className="w-full bg-slate-900/80 border border-slate-800 rounded-lg pl-9 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500"
            />
          </div>

          <div className="flex items-center gap-1.5 w-full md:w-auto overflow-x-auto pb-1">
            {[
              { id: 'all', label: 'All Routes' },
              { id: 'runbook', label: 'Private Runbooks' },
              { id: 'web', label: 'Internet Search' },
              { id: 'general', label: 'General Knowledge' },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setRouteFilter(tab.id)}
                className={`px-3 py-1 text-xs rounded-md font-medium whitespace-nowrap transition-all ${
                  routeFilter === tab.id
                    ? 'bg-emerald-600 text-white shadow-sm'
                    : 'bg-slate-900 text-slate-400 hover:text-white border border-slate-800'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Audit Records List */}
        {error && (
          <div className="p-4 rounded-xl bg-red-950/40 border border-red-800 text-red-300 text-xs">
            {error}
          </div>
        )}

        {filteredAudits.length === 0 ? (
          <div className="text-center py-16 border border-dashed border-slate-800 rounded-2xl">
            <ShieldCheck className="w-10 h-10 text-slate-600 mx-auto mb-2" />
            <p className="text-sm text-slate-300 font-medium">No audit logs found</p>
            <p className="text-xs text-slate-500 mt-1">
              Ask questions in the Incident Copilot chat to generate verification audits.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {filteredAudits.map((item) => {
              const isExpanded = expandedId === item.id;
              let traceList: string[] = [];
              let sourcesList: any[] = [];
              try {
                traceList = JSON.parse(item.trace_json || '[]');
              } catch (e) {}
              try {
                sourcesList = JSON.parse(item.sources_json || '[]');
              } catch (e) {}

              return (
                <div
                  key={item.id}
                  className="rounded-xl border border-slate-800/90 bg-slate-900/50 hover:border-slate-700/80 transition-all overflow-hidden"
                >
                  <div
                    onClick={() => setExpandedId(isExpanded ? null : item.id)}
                    className="p-4 cursor-pointer flex flex-col md:flex-row md:items-center justify-between gap-3"
                  >
                    <div className="space-y-1.5 flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="font-mono text-[10px] text-slate-400 bg-slate-800 px-1.5 py-0.5 rounded">
                          #{item.id}
                        </span>

                        <span
                          className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                            item.route?.includes('Runbook')
                              ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                              : item.route?.includes('Internet')
                              ? 'bg-blue-950 text-blue-300 border border-blue-800'
                              : 'bg-amber-950 text-amber-300 border border-amber-800'
                          }`}
                        >
                          {item.route || 'Direct'}
                        </span>

                        <span
                          className={`text-[10px] px-1.5 py-0.5 rounded ${
                            item.support_status === 'fully_supported'
                              ? 'bg-green-950 text-green-300 border border-green-800'
                              : item.support_status === 'partially_supported'
                              ? 'bg-yellow-950 text-yellow-300 border border-yellow-800'
                              : 'bg-slate-800 text-slate-400'
                          }`}
                        >
                          {item.support_status || 'No Grounding'}
                        </span>

                        {item.used_web === 1 && (
                          <span className="text-[10px] bg-purple-950 text-purple-300 border border-purple-800 px-1.5 py-0.5 rounded">
                            Web Active
                          </span>
                        )}

                        <span className="text-[10px] text-slate-500 flex items-center gap-1 ml-auto md:ml-0">
                          <Clock className="w-3 h-3" />
                          {new Date(item.created_at).toLocaleString()}
                        </span>
                      </div>

                      <h3 className="font-medium text-slate-200 text-xs md:text-sm line-clamp-1">
                        {item.question}
                      </h3>
                      <p className="text-slate-400 text-xs line-clamp-2 leading-relaxed">
                        {item.answer}
                      </p>
                    </div>

                    <div className="flex items-center space-x-2 text-slate-400 self-end md:self-center">
                      <span className="text-[11px] text-slate-500 font-mono">
                        {traceList.length} steps
                      </span>
                      {isExpanded ? (
                        <ChevronUp className="w-4 h-4 text-emerald-400" />
                      ) : (
                        <ChevronDown className="w-4 h-4" />
                      )}
                    </div>
                  </div>

                  {/* Expanded Detail View */}
                  {isExpanded && (
                    <div className="p-4 border-t border-slate-800/90 bg-slate-950/70 space-y-4 text-xs">
                      <div>
                        <span className="text-slate-400 font-semibold uppercase text-[10px]">
                          Complete Answer
                        </span>
                        <div className="mt-1 p-3 rounded-lg bg-slate-900 border border-slate-800 text-slate-200 whitespace-pre-wrap leading-relaxed">
                          {item.answer}
                        </div>
                      </div>

                      {/* Step Execution Trace */}
                      {traceList.length > 0 && (
                        <div>
                          <span className="text-slate-400 font-semibold uppercase text-[10px]">
                            Step-By-Step Execution Trace ({traceList.length} Steps)
                          </span>
                          <div className="mt-1 space-y-1">
                            {traceList.map((step, idx) => (
                              <div
                                key={idx}
                                className="flex items-center gap-2 p-1.5 rounded bg-slate-900/60 border border-slate-800/70 text-[11px] text-slate-300 font-mono"
                              >
                                <span className="text-emerald-500 text-[10px]">[{idx + 1}]</span>
                                <span>{step}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Sources */}
                      {sourcesList.length > 0 && (
                        <div>
                          <span className="text-slate-400 font-semibold uppercase text-[10px]">
                            Grounded Sources ({sourcesList.length})
                          </span>
                          <div className="mt-1 grid grid-cols-1 md:grid-cols-2 gap-2">
                            {sourcesList.map((src, idx) => (
                              <div
                                key={idx}
                                className="p-2 rounded bg-slate-900 border border-slate-800 text-[11px]"
                              >
                                <div className="font-medium text-slate-200 truncate">
                                  {src.title || src.source}
                                </div>
                                <div className="text-slate-500 truncate text-[10px] font-mono mt-0.5">
                                  {src.url || src.source}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};
