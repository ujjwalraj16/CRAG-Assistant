import { useCallback, useEffect, useState } from 'react';
import { useHealth } from './hooks/useHealth';
import { SidebarLayout } from './layouts/SidebarLayout';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { ChatPage } from './pages/ChatPage';
import { KnowledgeBasePage } from './pages/KnowledgeBasePage';
import { PipelinePage } from './pages/PipelinePage';
import { SettingsPage } from './pages/SettingsPage';
import { api } from './services/api';

export default function App() {
  const [activePage, setActivePage] = useState('chat');
  const [conversations, setConversations] = useState([]);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [selectedQueryId, setSelectedQueryId] = useState(null);

  const { health, refreshHealth } = useHealth();

  const refreshConversations = useCallback(async () => {
    try {
      const list = await api.listConversations();
      setConversations(list);
    } catch {
      // Backend might be booting; leave current state intact.
    }
  }, []);

  useEffect(() => {
    refreshConversations();
  }, [refreshConversations]);

  const handleSelectConversation = async (conversationId) => {
    setActiveConversationId(conversationId);
    try {
      const detail = await api.getConversation(conversationId);
      setMessages(detail.messages || []);
      const lastAssistant = [...(detail.messages || [])]
        .reverse()
        .find((m) => m.role === 'assistant' && m.query_id);
      if (lastAssistant?.query_id) {
        setSelectedQueryId(lastAssistant.query_id);
      }
    } catch {
      setMessages([]);
    }
  };

  const handleNewChat = () => {
    setActiveConversationId(null);
    setMessages([]);
  };

  const handleClearActiveChat = async () => {
    if (activeConversationId) {
      try {
        await api.deleteConversation(activeConversationId);
      } catch {
        // Ignore if already deleted.
      }
      await refreshConversations();
    }
    setActiveConversationId(null);
    setMessages([]);
  };

  const handleDeleteConversation = async (conversationId) => {
    try {
      await api.deleteConversation(conversationId);
      if (conversationId === activeConversationId) {
        setActiveConversationId(null);
        setMessages([]);
      }
      await refreshConversations();
    } catch {
      // Ignore deletion errors cleanly.
    }
  };

  const handleConversationCreated = (newConversationId) => {
    setActiveConversationId(newConversationId);
    refreshConversations();
  };

  const handleInspectPipeline = (queryId) => {
    setSelectedQueryId(queryId);
    setActivePage('pipeline');
  };

  return (
    <SidebarLayout
      activePage={activePage}
      onNavigate={setActivePage}
      conversations={conversations}
      activeConversationId={activeConversationId}
      onSelectConversation={handleSelectConversation}
      onNewChat={handleNewChat}
      onDeleteConversation={handleDeleteConversation}
      health={health}
    >
      {activePage === 'chat' && (
        <ChatPage
          conversationId={activeConversationId}
          messages={messages}
          onMessagesChange={setMessages}
          onConversationCreated={handleConversationCreated}
          onInspectPipeline={handleInspectPipeline}
          onKnowledgeUpdated={refreshHealth}
          onClearActiveChat={handleClearActiveChat}
        />
      )}

      {activePage === 'pipeline' && (
        <PipelinePage
          selectedQueryId={selectedQueryId}
          onSelectQueryId={setSelectedQueryId}
        />
      )}

      {activePage === 'knowledge' && (
        <KnowledgeBasePage onKnowledgeChanged={refreshHealth} />
      )}

      {activePage === 'analytics' && <AnalyticsPage />}

      {activePage === 'settings' && (
        <SettingsPage health={health} onSettingsSaved={refreshHealth} />
      )}
    </SidebarLayout>
  );
}

