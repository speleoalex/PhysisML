# PhysisML Chat Web UI

*Read this in: [Italiano](README.it.md)*

Minimal web interface to chat with `standalone/model.pt`, with admin login,
per-message feedback, and model backup/training.

```
webui/
├── deploy_local.sh       # start/stop/status (ports 8001 / 5501)
├── chat_server/          # FastAPI + SQLite + JWT
└── chat_webclient/       # HTML + vanilla JS + Bootstrap (CDN)
```

## Quick start

```bash
cd standalone/webui
./deploy_local.sh start
# backend  : http://localhost:8001   (docs: /docs)
# frontend : http://localhost:5501
```

On first run it creates a virtualenv in `chat_server/venv/`, a SQLite database
in `chat_server/data/chat.db`, and a default admin from `chat_server/.env`
(`admin@physisml.local` / `admin123` — change it).

## Usage

- **Chat**: open to everyone, no login needed. Every reply has an ID.
- **Login**: top-right button, admin credentials from `.env`.
- **Feedback**: admin only. A "feedback" button next to each reply (including
  older ones from history). Rating `+++/++/+/=/-` plus an optional
  "corrected answer" field used by training.
- **Backup**: copies `model.pt` + `tokenizer.json` to `chat_server/backups/<timestamp>/`.
- **Export**: downloads all collected feedback as JSONL.
- **Train now**: auto-backup, then applies `N` gradient steps per
  (prompt, corrected answer) pair rated `>= +`, saves the new `model.pt` and
  hot-swaps the in-memory model. Applied feedback is marked with `applied_at`.
  One training job at a time.

## API

All responses use the `{success, data, error}` envelope.

| Endpoint | Auth | Notes |
|---|---|---|
| `POST /api/v1/auth/login` | — | returns access token (default 24h) |
| `GET  /api/v1/auth/me` | user | current user info |
| `POST /api/v1/chat/send` | — | send a prompt, returns `message` |
| `GET  /api/v1/chat/history` | — | history (filterable by `session_id`) |
| `POST /api/v1/feedback` | admin | create feedback on a message |
| `PATCH /api/v1/feedback/{id}` | admin | edit existing feedback |
| `DELETE /api/v1/feedback/{id}` | admin | delete feedback |
| `POST /api/v1/admin/backup` | admin | snapshot into `backups/<ts>/` |
| `GET  /api/v1/admin/feedback/export` | admin | feedback as JSONL |
| `POST /api/v1/admin/train` | admin | start Train now (background) |
| `GET  /api/v1/admin/train/status` | admin | job status |
| `GET  /api/v1/admin/model/info` | admin | model parameters |

## Configuration

Variables in `chat_server/.env` (copied from `.env.example` on first run):

- `MODEL_PATH`, `TOKENIZER_PATH` — relative to `chat_server/`; default points to
  `standalone/model.pt` and `standalone/tokenizer.json`.
- `ADMIN_EMAIL`, `ADMIN_PASSWORD` — credentials created on first run.
- `JWT_SECRET` — **change it** before any real use.
- `CORS_ORIGINS` — comma-separated origins. The default already includes
  `http://localhost` (port 80) for Apache/XAMPP setups.
- `TRAIN_LR`, `TRAIN_STEPS_PER_SAMPLE` — Train now parameters.

### Serving the frontend from Apache/XAMPP

Symlink `chat_webclient/` into your web root (e.g. `~/public_html/`) and keep the
backend running on port 8001 (`./deploy_local.sh backend`). The default
`CORS_ORIGINS` already covers `http://localhost`.

## How Train now works

1. Copy `model.pt` and `tokenizer.json` to `backups/<timestamp>/`.
2. Load a fresh copy of the model in `train()` mode.
3. For each unapplied feedback rated `{+++, ++, +}` with a non-empty
   `corrected_text`: tokenize `prompt + "\n" + corrected_text` and run
   `train_step()` at a low LR (default `1e-5`) for `N` steps.
4. Save the new weights to `model.pt` and hot-swap the inference model
   (in-flight requests keep using the previous copy — no corruption).
5. Mark applied feedback with `applied_at`.

If a training run makes the model worse: stop the server, copy
`model.pt`/`tokenizer.json` back from the latest `backups/` folder, restart.

## Dependencies

- Python ≥ 3.10
- `torch`, `numpy` (installed automatically if missing)
- Reuses `standalone/physisml/` via `sys.path`.
