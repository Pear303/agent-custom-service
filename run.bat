@echo off
REM Set code page to UTF-8 to support Chinese characters
chcp 65001 >nul

REM Determine Python executable
set PYTHON_CMD=python
if exist ".venv\Scripts\python.exe" (
    set PYTHON_CMD=.venv\Scripts\python.exe
    echo [INFO] Found virtual environment.
) else (
    echo [WARN] No virtual environment found. Using global Python.
)

REM Install dependencies
echo [INFO] Installing dependencies from requirements.txt...
%PYTHON_CMD% -m pip install -r requirements.txt

echo ========================================
echo   OPC Smart Customer Service System
echo ========================================
echo.
echo   URL: http://localhost:8080
echo   API Docs: http://localhost:8080/docs
echo   Health Check: http://localhost:8080/health
echo.
echo   Press Ctrl+C to stop the server.
echo ========================================
echo.

REM Start the FastAPI server
REM --reload-dir 监控 api/ 和 agent/ 目录，AI 生成文件在 data/ 下不会触发重启
%PYTHON_CMD% -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload --reload-dir api --reload-dir agent

pause
