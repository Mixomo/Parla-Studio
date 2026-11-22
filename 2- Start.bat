@echo off
rem Parla Studio - start (run "1- Install.bat" first)
rem This console stays open next to the app and shows the log of every tab.
setlocal
cd /d "%~dp0"
title Parla Studio - console
if not exist ".venv\Scripts\python.exe" (
  echo [X] Not installed yet. Run "1- Install.bat" first.
  pause
  exit /b 1
)
rem Hugging Face models inside the project folder (portable)
set "HF_HOME=%~dp0models"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
".venv\Scripts\python.exe" -m parla_studio %*
if errorlevel 1 (
  echo.
  echo [X] Parla Studio closed with an error. See the lines above or data\app.log
  pause
)
