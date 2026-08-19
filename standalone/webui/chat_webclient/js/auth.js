const Auth = {
  async login(email, password) {
    const res = await API.post('/auth/login', { email, password });
    this.setAccessToken(res.data.access_token);
    this.setUser(res.data.user);
    return res.data.user;
  },

  logout() {
    this.clear();
  },

  onUnauthorized() {
    if (this.getAccessToken()) {
      this.clear();
      UI.toast('Sessione scaduta', 'warning');
      UI.refreshAuthState();
    }
  },

  getAccessToken() { return localStorage.getItem(CONFIG.STORAGE_KEYS.ACCESS_TOKEN); },
  setAccessToken(t) { localStorage.setItem(CONFIG.STORAGE_KEYS.ACCESS_TOKEN, t); },

  getUser() {
    const raw = localStorage.getItem(CONFIG.STORAGE_KEYS.USER);
    return raw ? JSON.parse(raw) : null;
  },
  setUser(u)  { localStorage.setItem(CONFIG.STORAGE_KEYS.USER, JSON.stringify(u)); },

  isAuthenticated() { return !!this.getAccessToken(); },
  isAdmin()         { return !!this.getUser()?.is_admin; },

  clear() {
    localStorage.removeItem(CONFIG.STORAGE_KEYS.ACCESS_TOKEN);
    localStorage.removeItem(CONFIG.STORAGE_KEYS.USER);
  },
};
