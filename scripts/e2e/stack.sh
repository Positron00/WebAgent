#!/usr/bin/env bash
# End-to-end stack for the Playwright tests (dev and CI only; nothing here is deployed).
#
#   scripts/e2e/stack.sh start    # fake Together, SMTP sink, backend, scheduler, built web app
#   scripts/e2e/stack.sh stop
#
# Everything runs on 127.0.0.1 with throwaway secrets in $E2E_DIR (default ./.e2e, git-ignored).
# No real model, search or mail service is used: Together is the repo's fake server, there is no
# Tavily key (web research degrades), and email goes to a local STARTTLS sink the backend trusts.
# The backend runs in dev without Git LFS weights: the guard and triage fall back, as allowed.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
E2E_DIR="${E2E_DIR:-$ROOT/.e2e}"
WEB_PORT="${E2E_WEB_PORT:-3100}"
BACKEND_PORT="${E2E_BACKEND_PORT:-8100}"
FAKE_PORT="${E2E_FAKE_PORT:-18210}"
SMTP_PORT="${E2E_SMTP_PORT:-2526}"
PY="${PYTHON:-python}"

wait_for() {  # url, name
  for _ in $(seq 1 120); do
    if curl -fs -o /dev/null "$1"; then return 0; fi
    sleep 2
  done
  echo "e2e: $2 did not come up ($1); last log lines:" >&2
  tail -n 40 "$E2E_DIR/logs/$2.log" >&2 || true
  exit 1
}

start() {
  mkdir -p "$E2E_DIR"/{logs,certs,mail,data}
  rm -f "$E2E_DIR"/mail/*.json
  # Fresh data each start: topics left by earlier runs reach max_active_topics (409 on create).
  rm -rf "${E2E_DIR:?}"/data/*
  umask 077
  if [ ! -f "$E2E_DIR/env" ]; then
    {
      echo "E2E_API_KEY=$(openssl rand -hex 32)"
      echo "E2E_SECRET_KEY=$(openssl rand -hex 48)"
      echo "E2E_SCHEDULER_TOKEN=$(openssl rand -hex 32)"
      echo "E2E_ACCESS_CODE=e2e-$(openssl rand -hex 8)"
      echo "E2E_SESSION_SECRET=$(openssl rand -hex 32)"
    } > "$E2E_DIR/env"
  fi
  # shellcheck disable=SC1091
  . "$E2E_DIR/env"
  umask 022
  # The sink's certificate lasts two days; renew it when it has less than an hour left.
  if ! openssl x509 -checkend 3600 -noout -in "$E2E_DIR/certs/cert.pem" >/dev/null 2>&1; then
    openssl req -x509 -newkey rsa:2048 -nodes -days 2 -subj "/CN=127.0.0.1" \
      -addext "subjectAltName=IP:127.0.0.1,DNS:localhost" \
      -keyout "$E2E_DIR/certs/key.pem" -out "$E2E_DIR/certs/cert.pem" 2>/dev/null
  fi

  # Every service: redirected as a whole and exec'd, so $! is the service itself and no wrapper shell
  # keeps this script's stdout open (a held pipe would hang whatever reads it, e.g. a CI step).
  (exec "$PY" "$ROOT/scripts/fake_together.py" --host 127.0.0.1 --port "$FAKE_PORT" --delay 0.1) \
    < /dev/null > "$E2E_DIR/logs/fake_together.log" 2>&1 & echo $! > "$E2E_DIR/fake_together.pid"
  (exec "$PY" "$ROOT/scripts/e2e/smtp_sink.py" --host 127.0.0.1 --port "$SMTP_PORT" --out "$E2E_DIR/mail" \
    --cert "$E2E_DIR/certs/cert.pem" --key "$E2E_DIR/certs/key.pem") \
    < /dev/null > "$E2E_DIR/logs/smtp_sink.log" 2>&1 & echo $! > "$E2E_DIR/smtp_sink.pid"

  local backend_env=(
    WEBAGENT_ENV=dev WEBAGENT_DATA_DIR="$E2E_DIR/data" WEBAGENT_SECURITY_API_KEY_ENABLED=true
    WEBAGENT_API_KEY="$E2E_API_KEY" WEBAGENT_SECURITY_SECRET_KEY="$E2E_SECRET_KEY"
    WEBAGENT_SCHEDULER_TOKEN="$E2E_SCHEDULER_TOKEN" WEBAGENT_BACKEND_URL="http://127.0.0.1:$BACKEND_PORT"
    TOGETHER_API_KEY="fake-$E2E_API_KEY" TOGETHER_API_BASE="http://127.0.0.1:$FAKE_PORT/v1" TAVILY_API_KEY=
    LANGSMITH_TRACING_ENABLED=false WEBAGENT_TRIAGE_COLLECT=false WEBAGENT_GUARD_COLLECT=false
    WEBAGENT_EMAIL_ENABLED=true WEBAGENT_EMAIL_TRANSPORT=smtp WEBAGENT_PUBLIC_URL="http://127.0.0.1:$WEB_PORT"
    WEBAGENT_SMTP_HOST=127.0.0.1 WEBAGENT_SMTP_PORT="$SMTP_PORT" WEBAGENT_SMTP_FROM=webagent@e2e.test
    SSL_CERT_FILE="$E2E_DIR/certs/cert.pem" WEBAGENT_SCHEDULER_ENABLED=true
    # Models fall back without weights; a Hugging Face download attempt once hung a run for ~5 minutes in CI.
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  )
  (cd "$ROOT/backend" && exec env "${backend_env[@]}" "$PY" -m uvicorn main:app --host 127.0.0.1 --port "$BACKEND_PORT") \
    < /dev/null > "$E2E_DIR/logs/backend.log" 2>&1 & echo $! > "$E2E_DIR/backend.pid"
  wait_for "http://127.0.0.1:$BACKEND_PORT/health" backend
  (cd "$ROOT/backend" && exec env "${backend_env[@]}" "$PY" -m app.scheduler) \
    < /dev/null > "$E2E_DIR/logs/scheduler.log" 2>&1 & echo $! > "$E2E_DIR/scheduler.pid"

  local web_env=(
    WEBAGENT_BACKEND_URL="http://127.0.0.1:$BACKEND_PORT" WEBAGENT_API_KEY="$E2E_API_KEY"
    WEBAGENT_ACCESS_CODE="$E2E_ACCESS_CODE" WEBAGENT_SESSION_SECRET="$E2E_SESSION_SECRET"
    WEBAGENT_DEFAULT_AGENTIC_MODE=always TOGETHER_API_KEY="placeholder-key-for-e2e-0123456789abcdef0123456789"
  )
  if [ "${E2E_SKIP_BUILD:-}" != "1" ]; then
    (cd "$ROOT" && exec env "${web_env[@]}" npm run -s build) < /dev/null > "$E2E_DIR/logs/web-build.log" 2>&1
  fi
  (cd "$ROOT" && exec env "${web_env[@]}" npx next start -H 127.0.0.1 -p "$WEB_PORT") \
    < /dev/null > "$E2E_DIR/logs/web.log" 2>&1 & echo $! > "$E2E_DIR/web.pid"
  wait_for "http://127.0.0.1:$WEB_PORT/login" web
  echo "e2e stack up: web http://127.0.0.1:$WEB_PORT, backend http://127.0.0.1:$BACKEND_PORT, mail $E2E_DIR/mail"
}

stop() {
  for name in web scheduler backend smtp_sink fake_together; do
    if [ -f "$E2E_DIR/$name.pid" ]; then
      kill "$(cat "$E2E_DIR/$name.pid")" 2>/dev/null || true
      rm -f "$E2E_DIR/$name.pid"
    fi
  done
  # next start forks a child server; make sure the port is free
  pkill -f "next start -H 127.0.0.1 -p $WEB_PORT" 2>/dev/null || true
  echo "e2e stack stopped"
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  *) echo "usage: $0 start|stop" >&2; exit 2 ;;
esac
