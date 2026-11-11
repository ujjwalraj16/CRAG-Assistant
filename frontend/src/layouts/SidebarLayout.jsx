import { useState } from 'react';
import {
  Activity,
  BarChart3,
  BookOpen,
  ChevronLeft,
  ChevronRight,
  Menu,
  MessageSquare,
  Plus,
  Settings,
  Sparkles,
  Trash2,
  X,
} from 'lucide-react';

const NAV_ITEMS = [
  { id: 'chat', label: 'Chat Assistant', icon: MessageSquare },
  { id: 'pipeline', label: 'CRAG Pipeline', icon: Activity },
  { id: 'knowledge', label: 'Knowledge Base', icon: BookOpen },
  { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  { id: 'settings', label: 'Settings', icon: Settings },
];

export function SidebarLayout({
  activePage,
  onNavigate,
  conversations,
  activeConversationId,
  onSelectConversation,
  onNewChat,
  onDeleteConversation,
  health,
  children,
}) {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  const ollamaOnline = Boolean(health?.ollama?.reachable);
  const vectorCount = health?.indexed_vectors ?? 0;

  const sidebarContent = (isMobile = false) => (
    <div className="flex h-full flex-col bg-[#0D1426] border-r border-slate-800/80">
      {/* Brand Header */}
      <div className="flex items-center justify-between border-b border-slate-800/80 px-4 py-4">
        <div className="flex items-center gap-2.5 overflow-hidden">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-[#22D3C5] to-[#8B5CF6] text-[#0B1020] shadow-lg shadow-[#22D3C5]/15">
            <Sparkles className="h-5 w-5" />
          </div>
          {(!collapsed || isMobile) && (
            <div className="min-w-0">
              <div className="truncate text-sm font-bold tracking-wide text-white">
                NEXUS AI
              </div>
              <div className="truncate text-[11px] font-medium text-[#22D3C5]">
                CRAG Adaptive Retrieval
              </div>
            </div>
          )}
        </div>

        {isMobile ? (
          <button
            type="button"
            onClick={() => setMobileOpen(false)}
            className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white"
          >
            <X className="h-5 w-5" />
          </button>
        ) : (
          <button
            type="button"
            onClick={() => setCollapsed((prev) => !prev)}
            className="hidden rounded-lg p-1.5 text-slate-400 transition hover:bg-slate-800 hover:text-white lg:inline-flex"
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
          </button>
        )}
      </div>

      {/* New Chat Button */}
      <div className="p-3">
        <button
          type="button"
          onClick={() => {
            onNewChat();
            onNavigate('chat');
            setMobileOpen(false);
          }}
          className="flex w-full items-center justify-center gap-2 rounded-xl bg-[#22D3C5] px-3 py-2.5 text-sm font-semibold text-[#0B1020] transition hover:bg-[#1ec2b5]"
          title="Start New Chat"
        >
          <Plus className="h-4 w-4 shrink-0" />
          {(!collapsed || isMobile) && <span>New Chat</span>}
        </button>
      </div>

      {/* Primary Navigation */}
      <nav className="space-y-1 px-3">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = activePage === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => {
                onNavigate(item.id);
                setMobileOpen(false);
              }}
              title={item.label}
              className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition ${
                isActive
                  ? 'bg-[#131B2F] text-[#22D3C5] border border-[#22D3C5]/25 shadow-sm'
                  : 'text-slate-300 hover:bg-[#131B2F]/70 hover:text-white border border-transparent'
              }`}
            >
              <Icon className="h-4 w-4 shrink-0" />
              {(!collapsed || isMobile) && <span className="truncate">{item.label}</span>}
            </button>
          );
        })}
      </nav>

      {/* Conversation History */}
      {(!collapsed || isMobile) && (
        <div className="mt-5 flex min-h-0 flex-1 flex-col border-t border-slate-800/70 px-3 pt-4">
          <div className="mb-2 px-1 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
            Recent Conversations
          </div>
          <div className="flex-1 space-y-1 overflow-y-auto pr-1">
            {conversations.length === 0 ? (
              <p className="px-2 py-3 text-xs text-slate-500">No conversations yet.</p>
            ) : (
              conversations.map((conv) => {
                const selected = conv.id === activeConversationId;
                return (
                  <div
                    key={conv.id}
                    className={`group flex items-center justify-between rounded-lg px-2.5 py-2 text-xs transition ${
                      selected
                        ? 'bg-[#131B2F] text-white'
                        : 'text-slate-400 hover:bg-[#131B2F]/60 hover:text-slate-200'
                    }`}
                  >
                    <button
                      type="button"
                      onClick={() => {
                        onSelectConversation(conv.id);
                        onNavigate('chat');
                        setMobileOpen(false);
                      }}
                      className="min-w-0 flex-1 truncate text-left"
                    >
                      {conv.title}
                    </button>
                    <button
                      type="button"
                      onClick={() => onDeleteConversation(conv.id)}
                      className="ml-2 hidden rounded p-1 text-slate-500 hover:bg-rose-500/15 hover:text-rose-300 group-hover:inline-flex"
                      title="Delete conversation"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                );
              })
            )}
          </div>
        </div>
      )}

      {/* System Status Footer */}
      <div className="mt-auto border-t border-slate-800/80 p-3">
        {(!collapsed || isMobile) ? (
          <div className="rounded-xl border border-slate-800 bg-[#131B2F]/80 p-3 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Ollama LLM</span>
              <span
                className={`inline-flex items-center gap-1.5 font-medium ${
                  ollamaOnline ? 'text-emerald-400' : 'text-amber-400'
                }`}
              >
                <span
                  className={`h-2 w-2 rounded-full ${
                    ollamaOnline ? 'bg-emerald-400' : 'bg-amber-400'
                  }`}
                />
                {ollamaOnline ? 'Online' : 'Offline'}
              </span>
            </div>
            <div className="mt-1.5 flex items-center justify-between text-slate-400">
              <span>Indexed Chunks</span>
              <span className="font-mono text-slate-200">{vectorCount}</span>
            </div>
          </div>
        ) : (
          <div className="flex justify-center" title={ollamaOnline ? 'Ollama Online' : 'Ollama Offline'}>
            <span
              className={`h-2.5 w-2.5 rounded-full ${
                ollamaOnline ? 'bg-emerald-400' : 'bg-amber-400'
              }`}
            />
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="flex h-screen w-full overflow-hidden bg-[#0B1020] text-slate-100">
      {/* Desktop Sidebar */}
      <aside
        className={`hidden shrink-0 transition-all duration-200 lg:block ${
          collapsed ? 'w-[74px]' : 'w-[272px]'
        }`}
      >
        {sidebarContent(false)}
      </aside>

      {/* Mobile Drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 flex lg:hidden">
          <div className="w-[280px] shrink-0">{sidebarContent(true)}</div>
          <div
            className="flex-1 bg-black/60 backdrop-blur-sm"
            onClick={() => setMobileOpen(false)}
          />
        </div>
      )}

      {/* Main Content Area */}
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        {/* Mobile Top Bar */}
        <header className="flex items-center justify-between border-b border-slate-800/80 bg-[#0D1426] px-4 py-3 lg:hidden">
          <button
            type="button"
            onClick={() => setMobileOpen(true)}
            className="rounded-lg p-1.5 text-slate-300 hover:bg-slate-800"
          >
            <Menu className="h-5 w-5" />
          </button>
          <div className="flex items-center gap-2 text-sm font-semibold text-white">
            <Sparkles className="h-4 w-4 text-[#22D3C5]" />
            <span>NEXUS AI · CRAG</span>
          </div>
          <div className="w-8" />
        </header>

        <main className="min-h-0 flex-1 overflow-y-auto">{children}</main>
      </div>
    </div>
  );
}

