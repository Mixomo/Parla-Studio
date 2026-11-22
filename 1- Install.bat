@echo off
rem ============================================================================
rem  Parla Studio - installer
rem  Everything (Python, packages, models) stays inside this folder. No global
rem  changes, no uv cache. Run it again at any time to repair or update.
rem ============================================================================
setlocal
cd /d "%~dp0"
title Parla Studio - Install
set "UV_NO_CACHE=1"
set "UV_PYTHON_INSTALL_DIR=%~dp0.runtime\python"
set "UV_PYTHON_PREFERENCE=only-managed"
set "UV_LINK_MODE=copy"
rem temporary unpack folder inside the project (PyTorch with CUDA takes several GB)
if not exist ".tmp" mkdir ".tmp"
set "TMP=%~dp0.tmp"
set "TEMP=%~dp0.tmp"

rem --- uv: use the one in PATH, or install a private copy in .runtime\uv
set "UV=uv"
where uv >nul 2>&1 && goto :have_uv
if exist "%~dp0.runtime\uv\uv.exe" goto :local_uv
echo [..] uv not found: installing a private copy in .runtime\uv
powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:UV_INSTALL_DIR='%~dp0.runtime\uv'; $env:UV_NO_MODIFY_PATH='1'; irm https://astral.sh/uv/install.ps1 | iex" || goto :error
:local_uv
set "UV=%~dp0.runtime\uv\uv.exe"
:have_uv

echo [..] Python 3.12
"%UV%" python install 3.12 || goto :error
echo [..] Packages (PyTorch CUDA 13.0, PySide6, models runtime)...
"%UV%" sync --frozen || "%UV%" sync || goto :error
if not exist "data" mkdir "data"
echo.
echo [OK] Ready. Run "2- Start.bat".
echo      AI models (several GB) are downloaded into models\ the first time each one is used.
pause
exit /b 0

:error
echo.
echo [X] Installation failed. Check your internet connection and run "1- Install.bat" again.
pause
exit /b 1
