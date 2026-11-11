const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const contentType = response.headers.get('content-type') || '';
  const data = contentType.includes('application/json')
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const message =
      (data && typeof data === 'object' && (data.detail || data.message)) ||
      (typeof data === 'string' && data) ||
      `Request failed with status ${response.status}`;
    throw new Error(message);
  }
  return data;
}

export const api = {
  // Health
  getHealth: () => request('/api/health'),

  // Chat & Conversations
  sendChat: (payload) =>
    request('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  listConversations: () => request('/api/chat/conversations'),
  getConversation: (conversationId) => request(`/api/chat/conversations/${conversationId}`),
  deleteConversation: (conversationId) =>
    request(`/api/chat/conversations/${conversationId}`, { method: 'DELETE' }),

  // Documents / Knowledge Base
  listDocuments: ({ search = '', category = '' } = {}) => {
    const params = new URLSearchParams();
    if (search) params.set('search', search);
    if (category && category !== 'All') params.set('category', category);
    const qs = params.toString();
    return request(`/api/documents${qs ? `?${qs}` : ''}`);
  },
  uploadDocument: (file, category = 'General') => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('category', category);
    return request('/api/documents/upload', {
      method: 'POST',
      body: formData,
    });
  },
  deleteDocument: (documentId) =>
    request(`/api/documents/${documentId}`, { method: 'DELETE' }),
  reindexDocument: (documentId) =>
    request(`/api/documents/${documentId}/reindex`, { method: 'POST' }),

  // Pipeline Traces
  listPipelineTraces: (limit = 25) => request(`/api/pipeline?limit=${limit}`),
  getPipelineTrace: (queryId) => request(`/api/pipeline/${queryId}`),

  // Analytics & Comparison
  getAnalyticsOverview: () => request('/api/analytics/overview'),
  getComparisonReport: () => request('/api/analytics/comparison'),
  runBenchmark: (payload = { top_k: 3, web_search_enabled: false, use_llm_judge: true }) =>
    request('/api/analytics/benchmark', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),

  // Settings
  getSettings: () => request('/api/settings'),
  updateSettings: (payload) =>
    request('/api/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
};

