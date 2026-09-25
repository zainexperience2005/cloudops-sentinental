import type { ChatResponse, AuditRecord, UploadResponse } from '../types';

// Supports both relative /api (via Vite proxy) and direct host fallback
const API_BASE = '/api';

export async function sendChatMessage(question: string, threadId: string): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      question: question.trim(),
      thread_id: threadId.trim(),
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Network error' }));
    throw new Error(errorData.detail || `Server returned ${response.status}`);
  }

  return response.json();
}

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Upload error' }));
    throw new Error(errorData.detail || `Upload failed with status ${response.status}`);
  }

  return response.json();
}

export async function fetchAudits(limit = 50): Promise<AuditRecord[]> {
  const response = await fetch(`${API_BASE}/audits?limit=${limit}`);
  if (!response.ok) {
    throw new Error(`Failed to load audits (${response.status})`);
  }
  return response.json();
}

export async function checkBackendHealth(): Promise<{ status: string; service: string }> {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) {
    throw new Error('Backend offline');
  }
  return response.json();
}
