/**
 * Minimal fetch wrapper. All responses follow
 * {success, data, error} envelope; on !success we throw.
 */
const API = {
  async request(endpoint, options = {}) {
    const url = `${CONFIG.API_URL}${endpoint}`;
    const headers = {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    };
    const token = Auth.getAccessToken();
    if (token) headers['Authorization'] = `Bearer ${token}`;

    const response = await fetch(url, { ...options, headers });
    return this._handle(response);
  },

  async _handle(response) {
    const ct = response.headers.get('content-type') || '';

    if (ct.includes('application/json')) {
      const data = await response.json();
      if (!response.ok || data.success === false) {
        const msg = data?.error?.message || data?.detail || `HTTP ${response.status}`;
        const err = new Error(msg);
        err.code   = data?.error?.code;
        err.status = response.status;
        if (response.status === 401) Auth.onUnauthorized();
        throw err;
      }
      return data;
    }

    // Text / jsonl
    const text = await response.text();
    if (!response.ok) {
      const err = new Error(`HTTP ${response.status}: ${text.slice(0, 200)}`);
      err.status = response.status;
      throw err;
    }
    return text;
  },

  get   (ep, params={})    { const q = new URLSearchParams(params).toString();
                             return this.request(q ? `${ep}?${q}` : ep); },
  post  (ep, body)         { return this.request(ep, { method: 'POST',
                             body: body === undefined ? undefined : JSON.stringify(body) }); },
  patch (ep, body)         { return this.request(ep, { method: 'PATCH',
                             body: JSON.stringify(body) }); },
  del   (ep)               { return this.request(ep, { method: 'DELETE' }); },

  // Raw fetch that returns text (for JSONL export)
  async getText(endpoint) {
    const url = `${CONFIG.API_URL}${endpoint}`;
    const headers = {};
    const token = Auth.getAccessToken();
    if (token) headers['Authorization'] = `Bearer ${token}`;
    const r = await fetch(url, { headers });
    if (!r.ok) {
      if (r.status === 401) Auth.onUnauthorized();
      throw new Error(`HTTP ${r.status}`);
    }
    return r.text();
  },
};
