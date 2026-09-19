const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
const cellsCache = new Map();
const cellDetailCache = new Map();

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || `Request failed with ${response.status}`);
  }
  return response.json();
}

export const api = {
  health: () => request('/api/health'),
  statistics: () => request('/api/statistics'),
  cells: ({ riskClass, limit = 100, offset = 0 } = {}) => {
    const params = new URLSearchParams({ limit, offset });
    if (riskClass) params.set('risk_class', riskClass);
    return request(`/api/cells?${params}`);
  },
  cell: (cellId) => {
    const cacheKey = String(cellId);
    if (cellDetailCache.has(cacheKey)) return Promise.resolve(cellDetailCache.get(cacheKey));
    return request(`/api/cells/${encodeURIComponent(cellId)}`).then((payload) => {
      cellDetailCache.set(cacheKey, payload);
      return payload;
    });
  },
  recommendations: (cellId) => request(`/api/cells/${encodeURIComponent(cellId)}/recommendations`),
  advisor: ({ cellId, question }) => request('/api/advisor', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ cell_id: cellId, question }),
  }),
};

export async function fetchAllCells({ riskClass } = {}) {
  const cacheKey = riskClass || 'all';
  if (cellsCache.has(cacheKey)) return cellsCache.get(cacheKey);
  const pageSize = 1000;
  const pages = [];
  let offset = 0;
  while (true) {
    const page = await api.cells({ riskClass, limit: pageSize, offset });
    pages.push(...page);
    if (page.length < pageSize) {
      cellsCache.set(cacheKey, pages);
      return pages;
    }
    offset += pageSize;
  }
}