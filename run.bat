@echo off
rem IK3 Smart Search (team 2): one-click start on Windows.
rem Creates .venv on first run, installs dependencies, starts the server on http://127.0.0.1:8000
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist .venv\Scripts\python.exe (
  echo [1/3] Creating virtual environment .venv ...
  py -3 -m venv .venv || python -m venv .venv || (echo Python 3.11+ not found. Install from python.org & exit /b 1)
)
echo [2/3] Installing dependencies ...
.venv\Scripts\python.exe -m pip install -q --disable-pip-version-check -r requirements.txt || exit /b 1
echo [3/3] Starting server: open http://127.0.0.1:8000  (API docs: /docs). Ctrl+C to stop.
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
