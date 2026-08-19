const Feedback = {
  _modal: null,
  _currentMessage: null,

  init() {
    this._modal = new bootstrap.Modal(document.getElementById('feedback-modal'));
    document.getElementById('feedback-form').addEventListener('submit', (e) => {
      e.preventDefault();
      this.submit();
    });
  },

  open(message) {
    this._currentMessage = message;
    document.getElementById('fb-msg-id').textContent = message.id;
    document.getElementById('fb-prompt').textContent = message.prompt;
    document.getElementById('fb-reply').textContent  = message.reply;

    const existing = (message.feedback || [])[0];
    const rating   = existing?.rating || '++';
    const text     = existing?.corrected_text || '';
    const radio    = document.querySelector(`input[name="fb-rating"][value="${rating}"]`);
    if (radio) radio.checked = true;
    document.getElementById('fb-corrected').value = text;

    document.getElementById('feedback-error').classList.add('d-none');
    this._modal.show();
  },

  async submit() {
    const rating   = document.querySelector('input[name="fb-rating"]:checked')?.value;
    const corrected = document.getElementById('fb-corrected').value.trim();
    const errBox   = document.getElementById('feedback-error');

    if (!rating) {
      errBox.textContent = 'Seleziona una valutazione.';
      errBox.classList.remove('d-none');
      return;
    }

    const msg = this._currentMessage;
    const existing = (msg.feedback || [])[0];

    try {
      let res;
      if (existing) {
        res = await API.patch(`/feedback/${existing.id}`, {
          rating,
          corrected_text: corrected || null,
        });
      } else {
        res = await API.post('/feedback', {
          message_id: msg.id,
          rating,
          corrected_text: corrected || null,
        });
      }
      Chat.updateMessageFeedback(msg.id, res.data);
      this._modal.hide();
      UI.toast('Feedback salvato', 'success');
    } catch (e) {
      errBox.textContent = e.message;
      errBox.classList.remove('d-none');
    }
  },
};
