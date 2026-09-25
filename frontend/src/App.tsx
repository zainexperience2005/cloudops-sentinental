import { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatView } from './components/ChatView';
import { AuditsView } from './components/AuditsView';
import { UploadView } from './components/UploadView';
import type { ThreadSession } from './types';
import { checkBackendHealth } from './services/api';

type ViewMode = 'chat' | 'audits' | 'upload';

export default function App() {
  // Simple URL path / hash detection for /audits, /upload, /chat
  const getInitialView = (): ViewMode => {
    const path = window.location.pathname.toLowerCase();
    if (path.includes('audit')) return 'audits';
    if (path.includes('upload')) return 'upload';
    return 'chat';
  };

  const [activeView, setActiveView] = useState<ViewMode>(getInitialView);
  const [backendHealthy, setBackendHealthy] = useState<boolean>(true);

  // Manage thread sessions
  const [threads, setThreads] = useState<ThreadSession[]>(() => {
    const saved = localStorage.getItem('sentinel_thread_sessions');
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch (e) {}
    }
    return [
      {
        id: 'incident-checkout-triage',
        title: 'Checkout API 500 Triage',
        lastUpdated: 'Just now',
        messageCount: 0,
      },
      {
        id: 'incident-payments-cpu',
        title: 'Payments High CPU',
        lastUpdated: 'Today',
        messageCount: 0,
      },
    ];
  });

  const [activeThreadId, setActiveThreadId] = useState<string>(() => {
    const savedLast = localStorage.getItem('sentinel_active_thread');
    return savedLast || 'incident-checkout-triage';
  });

  // Save thread sessions
  useEffect(() => {
    localStorage.setItem('sentinel_thread_sessions', JSON.stringify(threads));
  }, [threads]);

  useEffect(() => {
    localStorage.setItem('sentinel_active_thread', activeThreadId);
  }, [activeThreadId]);

  // Sync browser URL with views
  const handleSelectView = (view: ViewMode) => {
    setActiveView(view);
    const newPath = view === 'chat' ? '/chat' : view === 'audits' ? '/audits' : '/upload';
    window.history.pushState(null, '', newPath);
  };

  // Check backend health periodically
  useEffect(() => {
    const pollHealth = async () => {
      try {
        await checkBackendHealth();
        setBackendHealthy(true);
      } catch (e) {
        setBackendHealthy(false);
      }
    };

    pollHealth();
    const interval = setInterval(pollHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleCreateNewThread = (customId?: string) => {
    const newId = customId || `inc-${Math.floor(1000 + Math.random() * 9000)}`;
    const newSession: ThreadSession = {
      id: newId,
      title: `Incident ${newId}`,
      lastUpdated: 'Just now',
      messageCount: 0,
    };

    setThreads((prev) => [newSession, ...prev.filter((t) => t.id !== newId)]);
    setActiveThreadId(newId);
    handleSelectView('chat');
  };

  const handleDeleteThread = (threadId: string) => {
    const filtered = threads.filter((t) => t.id !== threadId);
    setThreads(filtered);
    localStorage.removeItem(`chat_history_${threadId}`);

    if (activeThreadId === threadId) {
      if (filtered.length > 0) {
        setActiveThreadId(filtered[0].id);
      } else {
        handleCreateNewThread('incident-default');
      }
    }
  };

  const handleUpdateThread = (threadId: string, messageCount: number) => {
    setThreads((prev) =>
      prev.map((t) =>
        t.id === threadId
          ? {
              ...t,
              messageCount,
              lastUpdated: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            }
          : t
      )
    );
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#090d16] text-slate-100 font-sans antialiased">
      {/* Left Sidebar */}
      <Sidebar
        activeView={activeView}
        onSelectView={handleSelectView}
        threads={threads}
        activeThreadId={activeThreadId}
        onSelectThread={setActiveThreadId}
        onNewThread={handleCreateNewThread}
        onDeleteThread={handleDeleteThread}
        backendHealthy={backendHealthy}
      />

      {/* Main View Area */}
      <main className="flex-1 flex flex-col min-w-0 h-full relative">
        {activeView === 'chat' && (
          <ChatView
            threadId={activeThreadId}
            onUpdateThread={handleUpdateThread}
            onNavigateToUpload={() => handleSelectView('upload')}
          />
        )}

        {activeView === 'audits' && <AuditsView />}

        {activeView === 'upload' && (
          <UploadView onUploadSuccess={() => handleSelectView('chat')} />
        )}
      </main>
    </div>
  );
}
