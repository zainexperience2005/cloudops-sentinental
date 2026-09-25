import React, { useState, useRef } from 'react';
import { 
  UploadCloud, 
  FileText, 
  CheckCircle, 
  AlertCircle, 
  Database, 
  Layers, 
  FileCheck,
  Cpu
} from 'lucide-react';
import { uploadDocument } from '../services/api';
import type { UploadResponse } from '../types';

interface Props {
  onUploadSuccess?: () => void;
}

export const UploadView: React.FC<Props> = ({ onUploadSuccess }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [result, setResult] = useState<UploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploadedHistory, setUploadedHistory] = useState<UploadResponse[]>(() => {
    const saved = localStorage.getItem('upload_history');
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch (e) {
        return [];
      }
    }
    return [
      { filename: 'checkout-api-runbook.md', chunks_indexed: 3, namespace: 'incident-runbooks' },
      { filename: 'deployment-rollback-sop.md', chunks_indexed: 2, namespace: 'incident-runbooks' },
      { filename: 'payments-high-cpu-runbook.md', chunks_indexed: 3, namespace: 'incident-runbooks' },
    ];
  });

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (!['pdf', 'txt', 'md', 'docx'].includes(ext || '')) {
      setError('Unsupported file type. Please upload a PDF, TXT, MD, or DOCX document.');
      return;
    }

    setIsUploading(true);
    setError(null);
    setResult(null);

    try {
      const data = await uploadDocument(file);
      setResult(data);
      const updated = [data, ...uploadedHistory];
      setUploadedHistory(updated);
      localStorage.setItem('upload_history', JSON.stringify(updated));
      if (onUploadSuccess) onUploadSuccess();
    } catch (err: any) {
      setError(err.message || 'Failed to ingest document into Pinecone');
    } finally {
      setIsUploading(false);
    }
  };

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const onDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = () => {
    setIsDragging(false);
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-[#090d16] overflow-hidden">
      {/* Header */}
      <header className="h-14 border-b border-slate-800/80 bg-slate-950/70 backdrop-blur-md px-6 flex items-center justify-between z-10">
        <div className="flex items-center space-x-3">
          <div className="p-1.5 rounded-lg bg-emerald-950/80 border border-emerald-800/60 text-emerald-400">
            <UploadCloud className="w-4 h-4" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-white">Ingest Operational Runbooks</h1>
            <p className="text-[11px] text-slate-400">Embed and index incident SOPs into the Pinecone Vector Database</p>
          </div>
        </div>

        <div className="flex items-center space-x-2 text-xs text-slate-400">
          <Database className="w-3.5 h-3.5 text-emerald-400" />
          <span className="font-mono">Namespace: incident-runbooks</span>
        </div>
      </header>

      {/* Main Upload Body */}
      <div className="flex-1 overflow-y-auto px-6 py-8 max-w-4xl mx-auto w-full space-y-6">
        {/* Drag and drop zone */}
        <div
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all ${
            isDragging
              ? 'border-emerald-500 bg-emerald-950/30 shadow-lg shadow-emerald-950/50'
              : 'border-slate-800 bg-slate-900/40 hover:border-slate-700 hover:bg-slate-900/60'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.txt,.md,.docx"
            className="hidden"
            onChange={(e) => {
              if (e.target.files && e.target.files.length > 0) {
                handleFile(e.target.files[0]);
              }
            }}
          />

          <div className="flex flex-col items-center justify-center space-y-3">
            <div className="p-4 rounded-2xl bg-emerald-950/50 border border-emerald-800/60 text-emerald-400">
              <UploadCloud className={`w-8 h-8 ${isUploading ? 'animate-bounce' : ''}`} />
            </div>

            <div>
              <h2 className="text-base font-semibold text-white">
                {isUploading ? 'Ingesting & Chunking Document...' : 'Click or drag SOP document here'}
              </h2>
              <p className="text-xs text-slate-400 mt-1">
                Supports standard incident documentation: <strong>PDF, Markdown (.md), DOCX, TXT</strong>
              </p>
            </div>

            <div className="flex items-center gap-2 pt-2 text-[10px] text-slate-400">
              <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700">Chunk size: 800</span>
              <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700">Overlap: 160</span>
              <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700">Dim: 3072</span>
            </div>
          </div>
        </div>

        {/* Success or Error banners */}
        {result && (
          <div className="p-4 rounded-xl bg-emerald-950/40 border border-emerald-800/80 text-emerald-300 text-xs flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <CheckCircle className="w-5 h-5 text-emerald-400 flex-shrink-0" />
              <div>
                <p className="font-semibold text-white">
                  Successfully ingested <span className="font-mono text-emerald-400">{result.filename}</span>
                </p>
                <p className="text-slate-400 mt-0.5">
                  Generated <strong>{result.chunks_indexed}</strong> semantic chunks in namespace{' '}
                  <strong className="text-emerald-300">{result.namespace}</strong>.
                </p>
              </div>
            </div>
            <span className="text-[10px] font-mono px-2 py-1 rounded bg-emerald-900/60 border border-emerald-700">
              Index Ready
            </span>
          </div>
        )}

        {error && (
          <div className="p-4 rounded-xl bg-red-950/40 border border-red-800/80 text-red-300 text-xs flex items-center space-x-3">
            <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
            <div>
              <p className="font-semibold text-white">Document Ingestion Failed</p>
              <p className="text-red-300/80 mt-0.5">{error}</p>
            </div>
          </div>
        )}

        {/* Information Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40 text-xs">
            <div className="flex items-center space-x-2 text-emerald-400 font-semibold mb-1">
              <Database className="w-4 h-4" />
              <span>Pinecone Vector Store</span>
            </div>
            <p className="text-slate-400 leading-relaxed">
              CloudOps Sentinel uses serverless Pinecone indexes for fast semantic vector search over private incident runbooks and architecture notes.
            </p>
          </div>

          <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40 text-xs">
            <div className="flex items-center space-x-2 text-cyan-400 font-semibold mb-1">
              <Cpu className="w-4 h-4" />
              <span>Embedding Engine</span>
            </div>
            <p className="text-slate-400 leading-relaxed">
              Embeddings are computed using OpenAI's <code className="text-cyan-300 font-mono">text-embedding-3-large</code> with 3072 dimensions for deep semantic representation.
            </p>
          </div>

          <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40 text-xs">
            <div className="flex items-center space-x-2 text-purple-400 font-semibold mb-1">
              <Layers className="w-4 h-4" />
              <span>Idempotent Ingestion</span>
            </div>
            <p className="text-slate-400 leading-relaxed">
              Chunks are tagged with stable SHA256 hashes so re-uploading updated documents replaces previous versions without creating duplicates.
            </p>
          </div>
        </div>

        {/* History of Ingested Documents */}
        <div>
          <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-3 flex items-center gap-1.5">
            <FileCheck className="w-3.5 h-3.5 text-emerald-400" />
            <span>Currently Active Runbooks</span>
          </h3>

          <div className="space-y-2">
            {uploadedHistory.map((doc, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between p-3 rounded-xl border border-slate-800 bg-slate-900/50 text-xs"
              >
                <div className="flex items-center space-x-2.5 min-w-0">
                  <FileText className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                  <span className="font-medium text-slate-200 truncate">{doc.filename}</span>
                </div>

                <div className="flex items-center space-x-3 text-[11px] text-slate-400 font-mono">
                  <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700">
                    {doc.chunks_indexed} chunks
                  </span>
                  <span className="text-emerald-400 hidden md:inline">
                    {doc.namespace}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
