# PhysisML Chat Web UI

*Leggi in: [English](README.md)*

Interfaccia web minimale per chattare con `standalone/model.pt`, con login
amministratore, feedback sui messaggi e backup/training del modello.

```text
webui/
├── deploy_local.sh       # start/stop/status (porte 8001 / 5501)
├── chat_server/          # FastAPI + SQLite + JWT
└── chat_webclient/       # HTML + vanilla JS + Bootstrap (CDN)
```

## Avvio rapido

```bash
cd standalone/webui
./deploy_local.sh start
# backend  : http://localhost:8001   (docs: /docs)
# frontend : http://localhost:5501
```

Al primo avvio vengono creati un virtualenv in `chat_server/venv/`, un database
SQLite in `chat_server/data/chat.db` e un admin di default preso da
`chat_server/.env` (`admin@physisml.local` / `admin123`, da cambiare).

## Uso

- **Chat**: aperta a tutti, anche senza login. Ogni risposta ha un ID.
- **Login**: bottone in alto a destra, credenziali admin dal `.env`.
- **Feedback**: solo da admin. Pulsante "feedback" accanto a ogni risposta
  (anche quelle vecchie dalla cronologia). Rating `+++/++/+/=/-` più un campo
  opzionale "risposta corretta" usato dal training.
- **Backup**: copia `model.pt` + `tokenizer.json` in `chat_server/backups/<timestamp>/`.
- **Export**: scarica tutto il feedback raccolto in JSONL.
- **Train now**: backup automatico, poi applica `N` gradient step su ogni coppia
  (prompt, risposta corretta) con rating `>= +`, salva il nuovo `model.pt` e
  scambia a caldo il modello in memoria. I feedback applicati vengono marcati
  con `applied_at`. Un solo training alla volta.

## API

Tutte le risposte usano l'envelope `{success, data, error}`.

| Endpoint | Auth | Note |
|---|---|---|
| `POST /api/v1/auth/login` | — | ritorna access token (default 24h) |
| `GET  /api/v1/auth/me` | user | info utente corrente |
| `POST /api/v1/chat/send` | — | manda un prompt, ritorna `message` |
| `GET  /api/v1/chat/history` | — | storia (filtrabile per `session_id`) |
| `POST /api/v1/feedback` | admin | crea feedback su un messaggio |
| `PATCH /api/v1/feedback/{id}` | admin | modifica feedback esistente |
| `DELETE /api/v1/feedback/{id}` | admin | elimina feedback |
| `POST /api/v1/admin/backup` | admin | snapshot in `backups/<ts>/` |
| `GET  /api/v1/admin/feedback/export` | admin | feedback in JSONL |
| `POST /api/v1/admin/train` | admin | avvia Train now (background) |
| `GET  /api/v1/admin/train/status` | admin | stato del job |
| `GET  /api/v1/admin/model/info` | admin | parametri del modello |

## Configurazione

Variabili in `chat_server/.env` (copiate da `.env.example` al primo avvio):

- `MODEL_PATH`, `TOKENIZER_PATH` — relativi a `chat_server/`; il default punta a
  `standalone/model.pt` e `standalone/tokenizer.json`.
- `ADMIN_EMAIL`, `ADMIN_PASSWORD` — credenziali create al primo avvio.
- `JWT_SECRET` — **da cambiare** prima di un uso reale.
- `CORS_ORIGINS` — lista di origin separati da virgola. Il default include già
  `http://localhost` (porta 80) per setup Apache/XAMPP.
- `TRAIN_LR`, `TRAIN_STEPS_PER_SAMPLE` — parametri del Train now.

### Servire il frontend con Apache/XAMPP

Crea un symlink di `chat_webclient/` nella web root (es. `~/public_html/`) e
tieni il backend attivo sulla porta 8001 (`./deploy_local.sh backend`). Il
`CORS_ORIGINS` di default copre già `http://localhost`.

## Come funziona il Train now

1. Copia `model.pt` e `tokenizer.json` in `backups/<timestamp>/`.
2. Carica una copia fresca del modello in modalità `train()`.
3. Per ogni feedback non applicato con rating ∈ `{+++, ++, +}` e
   `corrected_text` non vuoto: tokenizza `prompt + "\n" + corrected_text` e
   chiama `train_step()` con LR basso (default `1e-5`) per `N` step.
4. Salva i nuovi pesi su `model.pt` e **sostituisce a caldo** il modello usato
   dall'inferenza (le richieste in volo usano ancora la copia precedente,
   nessuna corruzione).
5. Marca i feedback applicati con `applied_at`.

Se un training peggiora il modello: ferma il server, ricopia
`model.pt`/`tokenizer.json` dall'ultima cartella in `backups/` e riavvia.

## Dipendenze

- Python ≥ 3.10
- `torch`, `numpy` (installati automaticamente se mancanti)
- Riusa `standalone/physisml/` tramite `sys.path`.
