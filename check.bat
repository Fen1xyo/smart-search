@echo off
rem Tests + quality evaluation (run after run.bat created .venv)
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist .venv\Scripts\python.exe (
  py -3 -m venv .venv || python -m venv .venv
  .venv\Scripts\python.exe -m pip install -q -r requirements.txt
)
.venv\Scripts\python.exe -m pytest -q || exit /b 1
.venv\Scripts\python.exe scripts\evaluate.py --sweep
