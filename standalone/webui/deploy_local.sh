#!/bin/bash
# PhysisML Chat Web UI — local deploy (backend FastAPI + static frontend).
# Mirrors the pattern of project_management/deploy_local.sh.
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$SCRIPT_DIR/chat_server"
CLIENT_DIR="$SCRIPT_DIR/chat_webclient"

BACKEND_PORT=8001
FRONTEND_PORT=5501
VENV_DIR="$SERVER_DIR/venv"

print_header() {
  echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
  echo -e "${BLUE}║            PhysisML Chat Web UI — local deploy             ║${NC}"
  echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
}

step() { echo -e "${GREEN}▶ $1${NC}"; }
ok()   { echo -e "${GREEN}✔ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠ $1${NC}"; }
err()  { echo -e "${RED}✖ $1${NC}"; }

check_python() {
  if command -v python3 &>/dev/null; then PY=python3
  elif command -v python &>/dev/null; then PY=python
  else err "Python not found"; exit 1; fi
  ok "Python $($PY --version 2>&1 | cut -d' ' -f2)"
}

setup_venv() {
  step "Virtualenv"
  if [ ! -d "$VENV_DIR" ]; then
    warn "Creating $VENV_DIR..."
    $PY -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  ok "venv active"
}

install_deps() {
  step "Installing dependencies"
  pip install --upgrade pip -q
  pip install -r "$SERVER_DIR/requirements.txt" -q
  ok "dependencies installed"
}

setup_env() {
  if [ ! -f "$SERVER_DIR/.env" ]; then
    warn "Creating .env from .env.example"
    cp "$SERVER_DIR/.env.example" "$SERVER_DIR/.env"
  fi
  mkdir -p "$SERVER_DIR/data" "$SERVER_DIR/backups"
}

kill_on_port() {
  local port=$1
  if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
    warn "Port $port busy, closing it..."
    kill $(lsof -Pi :$port -sTCP:LISTEN -t) 2>/dev/null || true
    sleep 1
  fi
}

start_backend() {
  step "Starting backend (port $BACKEND_PORT)"
  kill_on_port $BACKEND_PORT
  cd "$SERVER_DIR"
  uvicorn app.main:app --host 0.0.0.0 --port $BACKEND_PORT --reload &
  echo $! > "$SERVER_DIR/.backend.pid"
  sleep 2
  if kill -0 "$(cat "$SERVER_DIR/.backend.pid")" 2>/dev/null; then
    ok "backend PID $(cat "$SERVER_DIR/.backend.pid")"
  else
    err "backend failed to start"; exit 1
  fi
}

start_frontend() {
  step "Starting frontend (port $FRONTEND_PORT)"
  kill_on_port $FRONTEND_PORT
  cd "$CLIENT_DIR"
  $PY -m http.server $FRONTEND_PORT &
  echo $! > "$CLIENT_DIR/.frontend.pid"
  sleep 1
  if kill -0 "$(cat "$CLIENT_DIR/.frontend.pid")" 2>/dev/null; then
    ok "frontend PID $(cat "$CLIENT_DIR/.frontend.pid")"
  else
    err "frontend failed to start"
  fi
}

show_info() {
  echo ""
  echo -e "${BLUE}─────────────────────────────────────────────────────────────${NC}"
  echo -e "  ${YELLOW}Backend :${NC}  http://localhost:$BACKEND_PORT"
  echo -e "  ${YELLOW}Docs    :${NC}  http://localhost:$BACKEND_PORT/docs"
  echo -e "  ${YELLOW}Frontend:${NC}  http://localhost:$FRONTEND_PORT"
  local env_admin
  env_admin=$(grep -E '^ADMIN_EMAIL=' "$SERVER_DIR/.env" | cut -d= -f2-)
  echo -e "  ${YELLOW}Admin   :${NC}  ${env_admin:-admin@physisml.local} (password in $SERVER_DIR/.env)"
  echo -e "${BLUE}─────────────────────────────────────────────────────────────${NC}"
  echo -e "  Stop: ${GREEN}./deploy_local.sh stop${NC}"
  echo ""
}

stop_services() {
  step "Stopping services"
  for pidfile in "$SERVER_DIR/.backend.pid" "$CLIENT_DIR/.frontend.pid"; do
    if [ -f "$pidfile" ]; then
      PID=$(cat "$pidfile")
      if kill -0 $PID 2>/dev/null; then
        kill $PID 2>/dev/null || true
        ok "stopped PID $PID"
      fi
      rm -f "$pidfile"
    fi
  done
  pkill -f "uvicorn app.main:app" 2>/dev/null || true
  ok "all stopped"
}

show_status() {
  for name in backend frontend; do
    if [ "$name" = backend ]; then pidfile="$SERVER_DIR/.backend.pid"; port=$BACKEND_PORT
    else                           pidfile="$CLIENT_DIR/.frontend.pid"; port=$FRONTEND_PORT; fi
    if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
      echo -e "  $name : ${GREEN}● running${NC} (PID $(cat "$pidfile")) — http://localhost:$port"
    else
      echo -e "  $name : ${RED}● stopped${NC}"
    fi
  done
}

usage() {
  cat <<EOF
Usage: $0 [command]
  start    (default) start backend+frontend
  stop     stop both
  restart  stop + start
  status   show status
  backend  backend only
  frontend frontend only
EOF
}

print_header

case "${1:-start}" in
  start)
    check_python; setup_venv; install_deps; setup_env
    start_backend; start_frontend; show_info
    echo -e "${YELLOW}Ctrl+C to stop${NC}"
    wait
    ;;
  stop)     stop_services ;;
  restart)  stop_services; sleep 1
            check_python; setup_venv; install_deps; setup_env
            start_backend; start_frontend; show_info
            echo -e "${YELLOW}Ctrl+C to stop${NC}"
            wait ;;
  status)   show_status ;;
  backend)  check_python; setup_venv; install_deps; setup_env
            start_backend
            echo -e "${YELLOW}Ctrl+C to stop${NC}"
            wait ;;
  frontend) check_python; start_frontend
            echo -e "${YELLOW}Ctrl+C to stop${NC}"
            wait ;;
  help|-h|--help) usage ;;
  *) err "unknown command: $1"; usage; exit 1 ;;
esac
