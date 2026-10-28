@echo off
setlocal enabledelayedexpansion

title NeuralFact - Streamlit Detection Project
color 0C

echo ===============================================================================
echo                NEURALFACT: LLM HALLUCINATION DETECTOR
echo                         [ Streamlit Project ]
echo ===============================================================================
echo.

cd /d "%~dp0"

if exist "venv\Scripts\streamlit.exe" (
    set "STREAMLIT_EXE=venv\Scripts\streamlit.exe"
    set "PYTHON_EXE=venv\Scripts\python.exe"
) else (
    set "STREAMLIT_EXE=streamlit"
    set "PYTHON_EXE=python"
)

echo [*] Project Root: %CD%
echo [*] Streamlit Binary: %STREAMLIT_EXE%
echo [*] Starting Streamlit server on http://localhost:8501 ...
echo [*] Press Ctrl+C in this terminal window to stop the project.
echo.

"%STREAMLIT_EXE%" run app/streamlit_app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Streamlit exited with error code %ERRORLEVEL%.
    pause
)
