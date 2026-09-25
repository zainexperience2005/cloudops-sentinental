export interface SourceItem {
  type: 'internal' | 'web';
  title: string;
  source: string;
  url?: string | null;
  page?: number | null;
}

export interface ChatResponse {
  answer: string;
  route: string;
  used_web_search: boolean;
  support_status: 'fully_supported' | 'partially_supported' | 'no_support' | string;
  usefulness: 'useful' | 'not_useful' | string;
  sources: SourceItem[];
  trace: string[];
  thread_id: string;
  memory_turns: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  responseData?: ChatResponse;
  isLoading?: boolean;
  error?: string;
}

export interface AuditRecord {
  id: number;
  created_at: string;
  question: string;
  answer: string;
  route: string;
  used_web: number;
  support_status: string;
  usefulness: string;
  trace_json: string;
  sources_json: string;
}

export interface UploadResponse {
  filename: string;
  chunks_indexed: number;
  namespace: string;
}

export interface ThreadSession {
  id: string;
  title: string;
  lastUpdated: string;
  messageCount: number;
}
