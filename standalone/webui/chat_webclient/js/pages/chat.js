const Chat = {
  // In-memory mirror of messages rendered in the log.
  messages: [],

  init() {
    document.getElementById('chat-form').addEventListener('submit', (e) => {
      e.preventDefault();
      this.send();
    });
    document.getElementById('chat-prompt').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.send();
      }
    });
    document.getElementById('btn-new-session').addEventListener('click', () => {
      this.newSession();
    });
  },

  currentSession() {
    let sid = localStorage.getItem(CONFIG.STORAGE_KEYS.SESSION_ID);
    if (!sid) {
      sid = 's_' + Math.random().toString(36).slice(2, 10);
      localStorage.setItem(CONFIG.STORAGE_KEYS.SESSION_ID, sid);
    }
    return sid;
  },

  newSession() {
    const sid = 's_' + Math.random().toString(36).slice(2, 10);
    localStorage.setItem(CONFIG.STORAGE_KEYS.SESSION_ID, sid);
    this.messages = [];
    document.getElementById('chat-log').innerHTML = '';
    UI.refreshHistory();
  },

  async send() {
    const ta     = document.getElementById('chat-prompt');
    const prompt = ta.value.trim();
    if (!prompt) return;

    const btn = document.getElementById('chat-send');
    btn.disabled = true;

    const placeholder = this._appendPending(prompt);

    try {
      const res = await API.post('/chat/send', {
        prompt,
        session_id: this.currentSession(),
      });
      const msg = res.data.message;
      this.messages.push(msg);
      this._replacePending(placeholder, msg);
      ta.value = '';
      UI.refreshHistory();
    } catch (err) {
      this._replacePendingError(placeholder, err.message);
    } finally {
      btn.disabled = false;
      ta.focus();
    }
  },

  async loadHistory(sessionId) {
    const res = await API.get('/chat/history', { session_id: sessionId, limit: 100 });
    this.messages = res.data || [];
    const log = document.getElementById('chat-log');
    log.innerHTML = '';
    this.messages.forEach((m) => {
      log.appendChild(this._renderUser(m.prompt));
      log.appendChild(this._renderBot(m));
    });
    log.scrollTop = log.scrollHeight;
  },

  _appendPending(prompt) {
    const log = document.getElementById('chat-log');
    log.appendChild(this._renderUser(prompt));
    const botBubble = document.createElement('div');
    botBubble.className = 'bubble bot';
    botBubble.innerHTML = '<em class="text-muted">...</em>';
    log.appendChild(botBubble);
    log.scrollTop = log.scrollHeight;
    return botBubble;
  },

  _replacePending(bubble, msg) {
    const newBot = this._renderBot(msg);
    bubble.replaceWith(newBot);
    const log = document.getElementById('chat-log');
    log.scrollTop = log.scrollHeight;
  },

  _replacePendingError(bubble, err) {
    bubble.className = 'bubble bot text-danger';
    bubble.textContent = 'Error: ' + err;
  },

  _renderUser(text) {
    const el = document.createElement('div');
    el.className = 'bubble user';
    el.textContent = text;
    return el;
  },

  _renderBot(msg) {
    const el = document.createElement('div');
    el.className = 'bubble bot';
    el.dataset.msgId = msg.id;

    const body = document.createElement('div');
    body.textContent = msg.reply || '(vuoto)';
    el.appendChild(body);

    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.appendChild(Object.assign(document.createElement('span'),
                                   { textContent: `#${msg.id}` }));
    const existing = (msg.feedback || [])[0];
    if (existing) {
      const badge = document.createElement('span');
      badge.className = 'fb-badge';
      badge.textContent = existing.rating;
      meta.appendChild(badge);
    }
    if (Auth.isAdmin()) {
      const btn = document.createElement('button');
      btn.className = 'fb-btn';
      btn.innerHTML = '<i class="bi bi-pencil-square"></i> feedback';
      btn.addEventListener('click', () => Feedback.open(msg));
      meta.appendChild(btn);
    }
    el.appendChild(meta);
    return el;
  },

  // Called after a feedback save to update the bubble badge.
  updateMessageFeedback(messageId, fb) {
    const m = this.messages.find((x) => x.id === messageId);
    if (m) m.feedback = [fb];
    const bubble = document.querySelector(`.bubble.bot[data-msg-id="${messageId}"]`);
    if (bubble && m) bubble.replaceWith(this._renderBot(m));
  },
};
