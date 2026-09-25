import React, { useState } from 'react';
import { 
  ShieldAlert, 
  MessageSquare, 
  ShieldCheck, 
  UploadCloud, 
  Plus, 
  Trash2, 
  Activity, 
  Hash, 
  ChevronRight
} from 'lucide-react';
import type { ThreadSession } from '../types';

interface Props {
  activeView: 'chat' | 'audits' | 'upload';
  onSelectView: (view: 'chat' | 'audits' | 'upload') => void;
  threads: ThreadSession[];
  activeThreadId: string;
  onSelectThread: (threadId: string) => void;
  onNewThread: (customId?: string) => void;
  onDeleteThread: (threadId: string) => void;
  backendHealthy: boolean;
}

export const Sidebar: React.FC<Props> = ({
  activeView,
  onSelectView,
  threads,
  activeThreadId,
  onSelectThread,
  onNewThread,
  onDeleteThread,
  backendHealthy,
}) => {
  const [isCreatingCustom, setIsCreatingCustom] = useState(false);
  const [customIdInput, setCustomIdInput] = useState('');

  const handleCreateCustom = (e: React.FormEvent) => {
    e.preventDefault();
    if (customIdInput.trim()) {
      onNewThread(customIdInput.trim());
      setCustomIdInput('');
      setIsCreatingCustom(false);
    }
  };

  return (
    <aside className="w-64 md:w-72 bg-[#06090f] border-r border-slate-800/80 flex flex-col h-full select-none flex-shrink-0">
      {/* Brand Header */}
      <div className="p-4 border-b border-slate-800/80">
        <div className="flex items-center space-x-2.5">
          <div className="h-8 w-8 rounded-lg bg-emerald-950 border border-emerald-700/60 flex items-center justify-center text-emerald-400 shadow-md shadow-emerald-950/60">
            <ShieldAlert className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-white tracking-tight flex items-center gap-1.5">
              <span>CloudOps Sentinel</span>
            </h1>
            <p className="text-[10px] text-slate-400 font-mono">Self-RAG Incident Engine</p>
          </div>
        </div>

        {/* New Incident Button */}
        <button
          onClick={() => onNewThread()}
          className="w-full mt-3 flex items-center justify-center gap-2 py-2 px-3 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-xs shadow-md shadow-emerald-950/50 transition-all"
        >
          <Plus className="w-4 h-4" />
          <span>New Incident Session</span>
        </button>
      </div>

      {/* Navigation Routes */}
      <div className="p-3 space-y-1 border-b border-slate-800/80">
        <button
          onClick={() => onSelectView('chat')}
          className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-all ${
            activeView === 'chat'
              ? 'bg-slate-800 text-emerald-400 border border-slate-700'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
          }`}
        >
          <div className="flex items-center space-x-2.5">
            <MessageSquare className="w-4 h-4" />
            <span>Incident Copilot (/chat)</span>
          </div>
          {activeView === 'chat' && <ChevronRight className="w-3.5 h-3.5 text-emerald-400" />}
        </button>

        <button
          onClick={() => onSelectView('audits')}
          className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-all ${
            activeView === 'audits'
              ? 'bg-slate-800 text-emerald-400 border border-slate-700'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
          }`}
        >
          <div className="flex items-center space-x-2.5">
            <ShieldCheck className="w-4 h-4" />
            <span>Incident Audits (/audits)</span>
          </div>
          {activeView === 'audits' && <ChevronRight className="w-3.5 h-3.5 text-emerald-400" />}
        </button>

        <button
          onClick={() => onSelectView('upload')}
          className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-all ${
            activeView === 'upload'
              ? 'bg-slate-800 text-emerald-400 border border-slate-700'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
          }`}
        >
          <div className="flex items-center space-x-2.5">
            <UploadCloud className="w-4 h-4" />
            <span>Ingest Runbooks (/upload)</span>
          </div>
          {activeView === 'upload' && <ChevronRight className="w-3.5 h-3.5 text-emerald-400" />}
        </button>
      </div>

      {/* Incident Thread IDs Section */}
      <div className="flex-1 flex flex-col min-h-0 p-3">
        <div className="flex items-center justify-between px-1 mb-2">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
            Active Thread IDs ({threads.length})
          </span>
          <button
            onClick={() => setIsCreatingCustom(!isCreatingCustom)}
            className="text-[10px] text-emerald-400 hover:underline"
          >
            {isCreatingCustom ? 'Cancel' : '+ Custom ID'}
          </button>
        </div>

        {isCreatingCustom && (
          <form onSubmit={handleCreateCustom} className="mb-2">
            <input
              type="text"
              value={customIdInput}
              onChange={(e) => setCustomIdInput(e.target.value)}
              placeholder="e.g. incident-sev1-prod"
              className="w-full bg-slate-900 border border-emerald-700/60 rounded px-2.5 py-1 text-xs text-white placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
              autoFocus
            />
          </form>
        )}

        {/* Scrollable Thread list */}
        <div className="flex-1 overflow-y-auto space-y-1 pr-1">
          {threads.length === 0 ? (
            <div className="text-slate-600 text-xs py-4 text-center italic">
              No active threads. Click New Incident above.
            </div>
          ) : (
            threads.map((thread) => {
              const isActive = activeThreadId === thread.id;
              return (
                <div
                  key={thread.id}
                  onClick={() => {
                    onSelectThread(thread.id);
                    onSelectView('chat');
                  }}
                  className={`group flex items-center justify-between p-2 rounded-lg text-xs cursor-pointer transition-all ${
                    isActive
                      ? 'bg-slate-800/90 text-white border border-slate-700'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/50 border border-transparent'
                  }`}
                >
                  <div className="flex items-center space-x-2 min-w-0">
                    <Hash className={`w-3.5 h-3.5 flex-shrink-0 ${isActive ? 'text-emerald-400' : 'text-slate-600'}`} />
                    <div className="min-w-0">
                      <p className="font-mono truncate font-medium text-[11px]">{thread.id}</p>
                      <p className="text-[10px] text-slate-500 truncate">{thread.lastUpdated}</p>
                    </div>
                  </div>

                  <div className="flex items-center space-x-1">
                    {thread.messageCount > 0 && (
                      <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-slate-950 text-slate-400">
                        {thread.messageCount}
                      </span>
                    )}

                    {threads.length > 1 && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteThread(thread.id);
                        }}
                        className="opacity-0 group-hover:opacity-100 p-1 text-slate-500 hover:text-red-400 transition-opacity"
                        title="Delete thread"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* System Status Footer */}
      <div className="p-3 border-t border-slate-800/80 bg-slate-950/60 text-xs space-y-2">
        <div className="flex items-center justify-between text-[11px]">
          <span className="text-slate-400 flex items-center gap-1.5">
            <Activity className="w-3.5 h-3.5 text-slate-400" />
            <span>FastAPI Backend</span>
          </span>
          <span className="flex items-center gap-1">
            <span
              className={`h-2 w-2 rounded-full ${
                backendHealthy ? 'bg-emerald-400 animate-pulse' : 'bg-red-500'
              }`}
            ></span>
            <span className={backendHealthy ? 'text-emerald-400 font-medium' : 'text-red-400'}>
              {backendHealthy ? 'Connected' : 'Offline'}
            </span>
          </span>
        </div>

        <div className="pt-1 border-t border-slate-800/60 flex items-center justify-between text-[10px] text-slate-500 font-mono">
          <span>Pinecone: 3072-dim</span>
          <span>Self-RAG v2</span>
        </div>
      </div>
    </aside>
  );
};
