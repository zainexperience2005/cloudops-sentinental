import React, { useState, useRef, useEffect } from 'react';
import { 
  Send, 
  Bot, 
  User, 
  Terminal, 
  Copy, 
  Check, 
  AlertCircle, 
  Sparkles, 
  Trash2,
  Cpu,
  Layers
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { ChatMessage } from '../types';
import { sendChatMessage } from '../services/api';
import { SelfRagFlowStepVisualizer } from './SelfRagFlowStepVisualizer';
import { SourcesViewer } from './SourcesViewer';

interface Props {
  threadId: string;
  onUpdateThread: (threadId: string, messageCount: number) => void;
  onNavigateToUpload: () => void;
}

const QUICK_PROMPTS = [
  {
    title: 'Checkout API 500 Rate',
    desc: 'Runbook triage for checkout service errors',
    query: 'Checkout API is throwing elevated 500 error rates. What are the immediate triage and mitigation steps?',
  },
  {
    title: 'Payments High CPU',
    desc: 'Investigate CPU spikes & memory leak',
    query: 'Payments service pods are reporting high CPU (>85%) and slow database queries. How do I diagnose and fix this?',
  },
  {
    title: 'Deployment Rollback SOP',
    desc: 'Roll back failed canary deployment',
    query: 'A production canary deployment failed health checks. What is the standard rollback procedure?',
  },
  {
    title: 'General Ops Question',
    desc: 'Test direct general knowledge routing',
    query: 'What is the conceptual difference between RTO and RPO in disaster recovery planning?',
  },
];

export const ChatView: React.FC<Props> = ({ threadId, onUpdateThread, onNavigateToUpload }) => {
  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    const saved = localStorage.getItem(`chat_history_${threadId}`);
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch (e) {
        return [];
      }
    }
    return [];
  });

  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [activePipelineStep, setActivePipelineStep] = useState<string>('');

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Sync messages per thread
  useEffect(() => {
    const saved = localStorage.getItem(`chat_history_${threadId}`);
    if (saved) {
      try {
        setMessages(JSON.parse(saved));
      } catch (e) {
        setMessages([]);
      }
    } else {
      setMessages([]);
    }
  }, [threadId]);

  // Persist messages
  useEffect(() => {
    localStorage.setItem(`chat_history_${threadId}`, JSON.stringify(messages));
    onUpdateThread(threadId, messages.length);
  }, [messages, threadId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading, activePipelineStep]);

  const handleCopy = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleClearHistory = () => {
    if (confirm('Clear message history for this thread?')) {
      setMessages([]);
      localStorage.removeItem(`chat_history_${threadId}`);
      onUpdateThread(threadId, 0);
    }
  };

  const handleSend = async (questionText?: string) => {
    const query = (questionText || input).trim();
    if (!query || isLoading) return;

    setInput('');

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: query,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    const pendingAssistantMessageId = `assist-${Date.now()}`;
    const assistantPlaceholder: ChatMessage = {
      id: pendingAssistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      isLoading: true,
    };

    setMessages((prev) => [...prev, userMessage, assistantPlaceholder]);
    setIsLoading(true);

    // Live pipeline step indicator progression
    const stepMessages = [
      'Evaluating session memory and query context...',
      'Deciding retrieval strategy: internal runbooks vs direct...',
      'Querying Pinecone vector store (namespace: incident-runbooks)...',
      'Grading document relevance & evaluating evidence...',
      'Synthesizing grounded response using LLM...',
      'Running self-critique hallucination check...',
      'Verifying usefulness & saving to SQLite checkpointer...',
    ];

    let stepIndex = 0;
    setActivePipelineStep(stepMessages[0]);
    const stepTimer = setInterval(() => {
      stepIndex++;
      if (stepIndex < stepMessages.length) {
        setActivePipelineStep(stepMessages[stepIndex]);
      }
    }, 1200);

    try {
      const response = await sendChatMessage(query, threadId);
      clearInterval(stepTimer);
      setActivePipelineStep('');

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === pendingAssistantMessageId
            ? {
                ...msg,
                content: response.answer,
                isLoading: false,
                responseData: response,
              }
            : msg
        )
      );
    } catch (err: any) {
      clearInterval(stepTimer);
      setActivePipelineStep('');

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === pendingAssistantMessageId
            ? {
                ...msg,
                isLoading: false,
                error: err.message || 'Failed to process request',
                content: `**Incident Triage Error:** ${err.message || 'An unexpected error occurred while running the Self-RAG agent.'}`,
              }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-[#090d16] overflow-hidden">
      {/* Top Incident Control Header */}
      <header className="h-14 border-b border-slate-800/80 bg-slate-950/70 backdrop-blur-md px-6 flex items-center justify-between z-10">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-500 animate-pulse-subtle"></span>
            <span className="text-xs font-mono text-slate-400">Thread:</span>
            <span className="text-xs font-mono font-semibold px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-emerald-400">
              {threadId}
            </span>
          </div>

          <div className="hidden md:flex items-center space-x-1.5 text-xs text-slate-500 pl-3 border-l border-slate-800">
            <Layers className="w-3.5 h-3.5 text-slate-400" />
            <span>Turn-Aware Self-RAG</span>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          {messages.length > 0 && (
            <button
              onClick={handleClearHistory}
              title="Clear Thread History"
              className="p-1.5 text-slate-400 hover:text-red-400 hover:bg-slate-900 rounded-md transition-colors"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}

          <button
            onClick={onNavigateToUpload}
            className="flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-md bg-emerald-600/20 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-600/30 transition-all"
          >
            <Cpu className="w-3.5 h-3.5" />
            <span>Add Runbooks</span>
          </button>
        </div>
      </header>

      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto px-4 md:px-8 py-6 space-y-6">
        {messages.length === 0 ? (
          <div className="max-w-3xl mx-auto py-8">
            <div className="text-center space-y-3 mb-8">
              <div className="inline-flex p-3 rounded-2xl bg-emerald-950/40 border border-emerald-800/60 text-emerald-400 mb-2 shadow-lg shadow-emerald-950/30">
                <Terminal className="w-8 h-8" />
              </div>
              <h1 className="text-2xl font-bold tracking-tight text-white">
                CloudOps Sentinel Copilot
              </h1>
              <p className="text-sm text-slate-400 max-w-lg mx-auto">
                Autonomous Self-Reflective RAG for cloud operations, production troubleshooting, and incident response. Every step is evaluated and verified against grounded evidence.
              </p>
            </div>

            {/* Quick Prompt Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-2xl mx-auto">
              {QUICK_PROMPTS.map((p, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSend(p.query)}
                  className="text-left p-3.5 rounded-xl border border-slate-800/90 bg-slate-900/40 hover:bg-slate-800/60 hover:border-slate-700 transition-all group flex flex-col justify-between"
                >
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-semibold text-slate-200 text-xs group-hover:text-emerald-400 transition-colors">
                        {p.title}
                      </span>
                      <Sparkles className="w-3 h-3 text-slate-500 group-hover:text-emerald-400" />
                    </div>
                    <p className="text-[11px] text-slate-400 leading-relaxed">{p.desc}</p>
                  </div>
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`max-w-3xl mx-auto flex gap-3 ${
                message.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              {message.role === 'assistant' && (
                <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-emerald-950/80 border border-emerald-800/70 flex items-center justify-center text-emerald-400 shadow-sm mt-0.5">
                  <Bot className="w-4 h-4" />
                </div>
              )}

              <div
                className={`flex-1 max-w-[85%] rounded-2xl p-4 transition-all ${
                  message.role === 'user'
                    ? 'bg-slate-800 text-slate-100 rounded-tr-none border border-slate-700 shadow-md ml-auto'
                    : 'bg-slate-900/80 text-slate-200 rounded-tl-none border border-slate-800/90 glass-panel shadow-lg'
                }`}
              >
                {/* Header for assistant */}
                {message.role === 'assistant' && !message.isLoading && (
                  <div className="flex items-center justify-between mb-2 pb-2 border-b border-slate-800 text-xs">
                    <div className="flex items-center space-x-2">
                      <span className="font-semibold text-slate-300 text-[11px]">
                        CloudOps Sentinel
                      </span>
                      {message.responseData?.route && (
                        <span
                          className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                            message.responseData.route === 'Private Runbooks'
                              ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                              : message.responseData.route === 'Internet Search'
                              ? 'bg-blue-950 text-blue-300 border border-blue-800'
                              : 'bg-amber-950 text-amber-300 border border-amber-800'
                          }`}
                        >
                          {message.responseData.route}
                        </span>
                      )}
                    </div>

                    <div className="flex items-center space-x-1">
                      <button
                        onClick={() => handleCopy(message.id, message.content)}
                        className="p-1 text-slate-400 hover:text-white rounded transition-colors"
                        title="Copy answer"
                      >
                        {copiedId === message.id ? (
                          <Check className="w-3.5 h-3.5 text-emerald-400" />
                        ) : (
                          <Copy className="w-3.5 h-3.5" />
                        )}
                      </button>
                    </div>
                  </div>
                )}

                {/* Loading State with live pipeline text */}
                {message.isLoading ? (
                  <div className="space-y-3 py-2">
                    <div className="flex items-center space-x-2.5 text-emerald-400 text-xs">
                      <div className="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></div>
                      <span className="font-medium animate-pulse">{activePipelineStep || 'Executing Self-RAG Graph...'}</span>
                    </div>

                    <div className="space-y-1.5">
                      <div className="h-2 bg-slate-800 rounded animate-pulse w-3/4"></div>
                      <div className="h-2 bg-slate-800 rounded animate-pulse w-5/6"></div>
                      <div className="h-2 bg-slate-800 rounded animate-pulse w-2/3"></div>
                    </div>
                  </div>
                ) : message.error ? (
                  <div className="flex items-start space-x-2 text-red-400 text-xs">
                    <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-semibold">Error running Self-RAG agent</p>
                      <p className="text-slate-400 mt-1">{message.error}</p>
                    </div>
                  </div>
                ) : (
                  <>
                    {/* Markdown Body */}
                    <div className="prose prose-invert prose-xs max-w-none text-slate-200 leading-relaxed break-words">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          code({ className, children, ...props }: any) {
                            return (
                              <code
                                className="px-1.5 py-0.5 rounded bg-slate-950 font-mono text-[11px] text-emerald-400 border border-slate-800"
                                {...props}
                              >
                                {children}
                              </code>
                            );
                          },
                          pre({ children }: any) {
                            return (
                              <pre className="p-3 my-2 rounded-lg bg-slate-950 border border-slate-800 overflow-x-auto text-[11px] font-mono text-slate-300">
                                {children}
                              </pre>
                            );
                          },
                          ul({ children }: any) {
                            return <ul className="list-disc pl-4 space-y-1 my-2">{children}</ul>;
                          },
                          ol({ children }: any) {
                            return <ol className="list-decimal pl-4 space-y-1 my-2">{children}</ol>;
                          },
                          h1({ children }: any) {
                            return <h1 className="text-base font-bold text-white mt-3 mb-1.5">{children}</h1>;
                          },
                          h2({ children }: any) {
                            return <h2 className="text-sm font-semibold text-white mt-2.5 mb-1">{children}</h2>;
                          },
                          h3({ children }: any) {
                            return <h3 className="text-xs font-semibold text-white mt-2 mb-1">{children}</h3>;
                          },
                        }}
                      >
                        {message.content}
                      </ReactMarkdown>
                    </div>

                    {/* Step-by-Step Flow Pipeline Visualizer */}
                    {message.responseData?.trace && message.responseData.trace.length > 0 && (
                      <SelfRagFlowStepVisualizer
                        trace={message.responseData.trace}
                        route={message.responseData.route}
                        supportStatus={message.responseData.support_status}
                        usefulness={message.responseData.usefulness}
                      />
                    )}

                    {/* Grounded Sources */}
                    {message.responseData?.sources && message.responseData.sources.length > 0 && (
                      <SourcesViewer sources={message.responseData.sources} />
                    )}
                  </>
                )}

                <div className="text-[10px] text-slate-500 mt-2 text-right">
                  {message.timestamp}
                </div>
              </div>

              {message.role === 'user' && (
                <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-300 shadow-sm mt-0.5">
                  <User className="w-4 h-4" />
                </div>
              )}
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Bar Area */}
      <div className="p-4 border-t border-slate-800/80 bg-slate-950/80 backdrop-blur-md">
        <div className="max-w-3xl mx-auto">
          <div className="relative rounded-xl border border-slate-800 bg-slate-900/70 focus-within:border-emerald-500/80 focus-within:ring-1 focus-within:ring-emerald-500/50 transition-all shadow-inner">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask an incident question (e.g. 'Checkout API 500 error triage steps')..."
              rows={2}
              className="w-full bg-transparent px-4 py-3 text-xs md:text-sm text-slate-100 placeholder-slate-500 focus:outline-none resize-none leading-relaxed"
              disabled={isLoading}
            />

            <div className="flex items-center justify-between px-3 pb-2.5">
              <span className="text-[10px] text-slate-500 font-mono">
                Press <kbd className="px-1 py-0.5 rounded bg-slate-800 text-slate-400">Enter</kbd> to send, <kbd className="px-1 py-0.5 rounded bg-slate-800 text-slate-400">Shift+Enter</kbd> for newline
              </span>

              <button
                onClick={() => handleSend()}
                disabled={!input.trim() || isLoading}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-600 text-white font-medium text-xs hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed transition-all shadow-md shadow-emerald-950/50"
              >
                <span>Send</span>
                <Send className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
