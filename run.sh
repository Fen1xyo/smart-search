#!/usr/bin/env sh
# Linux/macOS: создаёт .venv, ставит зависимости, запускает сервер на http://127.0.0.1:8000
set -e
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/python -m pip install -q --disable-pip-version-check -r requirements.txt
exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port "${PORT:-8000}"
