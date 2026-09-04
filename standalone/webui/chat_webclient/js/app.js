/**
 * App bootstrap + shared UI helpers (toast, header state, history sidebar,
 * admin actions: Backup / Train now / Export).
 */
const UI = {
  _loginModal: null,
  _trainTimer: null,

  init() {
    this._loginModal = new bootstrap.Modal(document.getElementById('login-modal'));

    document.getElementById('btn-login').addEventListener('click', () => {
      document.getElementById('login-error').classList.add('d-none');
      this._loginModal.show();
    });

    document.getElementById('login-form').addEventListener('submit', async (e) => {
      e.preventDefault();
      const email = document.getElementById('login-email').value.trim();
      const pw    = document.getElementById('login-password').value;
      const err   = document.getElementById('login-error');
      try {
        await Auth.login(email, pw);
        this._loginModal.hide();
        UI.toast('Login ok', 'success');
        this.refreshAuthState();
      } catch (ex) {
        err.textContent = ex.message;
        err.classList.remove('d-none');
      }
    });

    document.getElementById('btn-logout').addEventListener('click', () => {
      Auth.logout();
      this.refreshAuthState();
      UI.toast('Logged out', 'info');
    });

    document.getElementById('btn-backup').addEventListener('click', () => this.doBackup());
    document.getElementById('btn-train') .addEventListener('click', () => this.doTrain());
    document.getElementById('btn-export').addEventListener('click', () => this.doExport());

    Feedback.init();
    Chat.init();

    this.refreshAuthState();
    this.refreshHistory();
    Chat.loadHistory(Chat.currentSession()).catch(() => {});
  },

  refreshAuthState() {
    const isAuth  = Auth.isAuthenticated();
    const isAdmin = Auth.isAdmin();

    document.getElementById('btn-login') .classList.toggle('d-none',  isAuth);
    document.getElementById('btn-logout').classList.toggle('d-none', !isAuth);
    document.getElementById('btn-backup').classList.toggle('d-none', !isAdmin);
    document.getElementById('btn-train') .classList.toggle('d-none', !isAdmin);
    document.getElementById('btn-export').classList.toggle('d-none', !isAdmin);

    const lbl = document.getElementById('user-label');
    lbl.textContent = isAuth ? Auth.getUser()?.email || '' : '';

    // Re-render current bubbles so feedback buttons appear/disappear
    Chat.loadHistory(Chat.currentSession()).catch(() => {});

    if (isAdmin) this._pollTrainStatus();
    else         this._stopTrainPolling();
  },

  async refreshHistory() {
    // Shows most recent 30 messages (grouped by session_id).
    try {
      const res = await API.get('/chat/history', { limit: 100 });
      const msgs = res.data || [];
      const bySession = new Map();
      msgs.forEach((m) => {
        if (!bySession.has(m.session_id)) bySession.set(m.session_id, []);
        bySession.get(m.session_id).push(m);
      });

      const list = document.getElementById('history-list');
      list.innerHTML = '';
      const current = Chat.currentSession();
      const isAdmin = Auth.isAdmin();
      [...bySession.entries()].forEach(([sid, ms]) => {
        const row = document.createElement('div');
        row.className = 'list-group-item' +
                        (sid === current ? ' active' : '');

        const body = document.createElement('div');
        body.className = 'hist-body';
        body.style.cursor = 'pointer';
        const preview = ms[ms.length - 1].prompt.slice(0, 40);
        body.innerHTML = `
          <div class="fw-semibold">${ms.length} msg</div>
          <div class="text-muted">${this._esc(preview)}</div>`;
        body.addEventListener('click', () => {
          localStorage.setItem(CONFIG.STORAGE_KEYS.SESSION_ID, sid);
          Chat.loadHistory(sid);
          this.refreshHistory();
        });
        row.appendChild(body);

        const tag = document.createElement('span');
        tag.className = 'text-muted small';
        tag.textContent = sid.slice(0, 6);
        row.appendChild(tag);

        if (isAdmin) {
          const del = document.createElement('button');
          del.className = 'btn btn-sm btn-outline-danger fb-btn ms-1';
          del.title = 'Delete chat';
          del.innerHTML = '<i class="bi bi-trash"></i>';
          del.addEventListener('click', (e) => {
            e.stopPropagation();
            this.deleteSession(sid, ms.length);
          });
          row.appendChild(del);
        }

        list.appendChild(row);
      });
    } catch (e) {
      console.warn('history failed:', e);
    }
  },

  async deleteSession(sid, count) {
    if (!confirm(`Delete chat ${sid.slice(0, 6)} (${count} messages)?\nThis cannot be undone.`)) return;
    try {
      await API.del(`/admin/sessions/${encodeURIComponent(sid)}`);
      UI.toast('Chat deleted', 'success');
      // If we just deleted the current session, reset the view.
      if (sid === Chat.currentSession()) {
        Chat.newSession();
      } else {
        this.refreshHistory();
      }
    } catch (e) {
      UI.toast('Delete failed: ' + e.message, 'danger');
    }
  },

  async doBackup() {
    const btn = document.getElementById('btn-backup');
    btn.disabled = true;
    try {
      const res = await API.post('/admin/backup', {});
      UI.toast(`Backup saved: ${res.data.path}`, 'success');
    } catch (e) {
      UI.toast('Backup failed: ' + e.message, 'danger');
    } finally {
      btn.disabled = false;
    }
  },

  async doTrain() {
    if (!confirm('Start a training run on the collected feedback?\n' +
                 'The current model is backed up automatically.')) return;
    try {
      const res = await API.post('/admin/train', {});
      UI.toast('Training started', 'info');
      this._showTrainBanner(res.data);
      this._pollTrainStatus();
    } catch (e) {
      UI.toast('Train now failed: ' + e.message, 'danger');
    }
  },

  async doExport() {
    try {
      const text = await API.getText('/admin/feedback/export');
      const blob = new Blob([text], { type: 'application/jsonl' });
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement('a');
      const ts   = new Date().toISOString().replace(/[:.]/g, '-');
      a.href = url; a.download = `feedback_${ts}.jsonl`;
      a.click();
      URL.revokeObjectURL(url);
      UI.toast('Feedback exported', 'success');
    } catch (e) {
      UI.toast('Export failed: ' + e.message, 'danger');
    }
  },

  _pollTrainStatus() {
    if (!Auth.isAdmin()) return;
    this._stopTrainPolling();
    const tick = async () => {
      try {
        const res = await API.get('/admin/train/status');
        this._showTrainBanner(res.data);
        if (res.data.state === 'running') {
          this._trainTimer = setTimeout(tick, CONFIG.TRAIN_POLL_MS);
        }
      } catch (e) {
        // ignore — probably logged out
      }
    };
    tick();
  },

  _stopTrainPolling() {
    if (this._trainTimer) { clearTimeout(this._trainTimer); this._trainTimer = null; }
  },

  _showTrainBanner(status) {
    const el = document.getElementById('train-banner');
    if (!status || status.state === 'idle') { el.classList.add('d-none'); return; }
    el.classList.remove('d-none');
    if (status.state === 'running') {
      const pct = Math.round((status.progress || 0) * 100);
      el.className = 'alert alert-info py-2 mb-0';
      el.textContent = `Training running: ${status.processed}/${status.total} (${pct}%)`
                     + (status.last_loss !== null && status.last_loss !== undefined
                        ? ` — loss ${status.last_loss.toFixed(4)}` : '');
    } else if (status.state === 'done') {
      el.className = 'alert alert-success py-2 mb-0';
      el.textContent = `Training complete. Backup: ${status.backup_path || '(n/a)'}`;
    } else if (status.state === 'error') {
      el.className = 'alert alert-danger py-2 mb-0';
      el.textContent = `Training error: ${status.error}`;
    }
  },

  toast(message, variant = 'info') {
    const wrap = document.getElementById('toast-container');
    const el = document.createElement('div');
    el.className = `toast align-items-center text-bg-${variant} border-0`;
    el.setAttribute('role', 'alert');
    el.innerHTML = `
      <div class="d-flex">
        <div class="toast-body">${this._esc(message)}</div>
        <button type="button" class="btn-close btn-close-white me-2 m-auto"
                data-bs-dismiss="toast"></button>
      </div>`;
    wrap.appendChild(el);
    const t = new bootstrap.Toast(el, { delay: 4000 });
    t.show();
    el.addEventListener('hidden.bs.toast', () => el.remove());
  },

  _esc(s) {
    return String(s ?? '').replace(/[&<>"']/g, (c) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
  },
};

document.addEventListener('DOMContentLoaded', () => UI.init());
