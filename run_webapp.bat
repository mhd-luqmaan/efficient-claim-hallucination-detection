@echo off
setlocal enabledelayedexpansion

title NeuralFact - Showcase Web Application
color 0C

echo ===============================================================================
echo                NEURALFACT: TECHNICAL SHOWCASE WEB APP
echo                          [ Netflix-Style UI ]
echo ===============================================================================
echo.

cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

set "PORT=8080"
set "URL=http://localhost:%PORT%"

echo [*] Web App Directory: %CD%\showcase
echo [*] Python Binary:     %PYTHON_EXE%
echo [*] Web Server Port:   %PORT%
echo [*] Opening %URL% in your default web browser...
echo.

start "" "%URL%"

echo [*] Showcase server is active.
echo [*] Press Ctrl+C in this terminal window to stop the web app.
echo ===============================================================================
echo.

"%PYTHON_EXE%" -m http.server %PORT% --directory showcase

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Server stopped or encountered an error (code: %ERRORLEVEL%).
    pause
)
