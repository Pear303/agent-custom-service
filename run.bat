@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

echo =========================================
echo   OPC Smart Customer Service System
echo =========================================
echo.

REM -- Python venv -----------------------------------------------
set PYTHON_CMD=python
if exist ".venv\Scripts\python.exe" (
    set PYTHON_CMD=.venv\Scripts\python.exe
    echo [OK] .venv found
) else (
    echo [WARN] No .venv, using system python
)

REM -- Dependencies ----------------------------------------------
echo [..] Installing dependencies...
%PYTHON_CMD% -m pip install -r requirements.txt -q 2>nul
if errorlevel 1 (
    echo [WARN] pip install had errors, continuing...
)

REM -- Mode detection --------------------------------------------
REM Default: start frontend + backend, then open browser
REM Set PROD_MODE=true to start backend only (production)
if "%PROD_MODE%"=="1"   set PROD_MODE=true
if "%PROD_MODE%"=="yes" set PROD_MODE=true
if /i "%PROD_MODE%"=="true" (goto :prod_mode) else (goto :dev_mode)

:dev_mode
echo.
echo   [DEV MODE] Starting frontend + backend...
echo.

REM -- Ensure npm dependencies installed -------------------------
if not exist "frontend\node_modules" (
    echo [..] npm install...
    cd frontend
    call npm install 2>nul
    cd ..
)

REM -- Start Vite frontend (new window) --------------------------
echo [..] Starting frontend Vite (http://localhost:5173)...
start "SmartCS Frontend" cmd /c "cd /d %~dp0frontend && npm run dev"

REM -- Wait a bit for Vite to start -------------------------------
timeout /t 3 /nobreak >nul

REM -- Start backend (new window) ---------------------------------
echo [..] Starting backend API (http://localhost:8080)...
start "SmartCS Backend" cmd /c "cd /d %~dp0 && %PYTHON_CMD% -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload --reload-dir=api --reload-dir=agent_by_langgraph --reload-dir=agent_core --no-access-log"

REM -- Open browser ----------------------------------------------
timeout /t 2 /nobreak >nul
echo [OK] Opening browser...
start http://localhost:5173

echo.
echo =========================================
echo   Services starting up, please wait...
echo.
echo   Frontend page : http://localhost:5173
echo   Backend API   : http://localhost:8080
echo   API Docs      : http://localhost:8080/docs
echo.
echo   Close frontend/backend windows to stop.
echo =========================================
echo.
goto :eof

:prod_mode
REM -- Check frontend build -------------------------------------
if exist "frontend\dist\index.html" goto :prod_ok

echo.
echo [WARN] frontend\dist\ not found (GET / will 404)
echo.
set BUILD=
set /p BUILD="Build frontend now? (npm must be in PATH) [Y/n] "
if /i "!BUILD!"=="n" goto :prod_skip_build

echo [..] npm install...
cd frontend
call npm install 2>nul
echo [..] vite build...
call npx vite build
cd ..
if exist "frontend\dist\index.html" (
    echo [OK] Build done
) else (
    echo [FAIL] Build failed - check Node.js >= 18 and npm
)

:prod_skip_build
:prod_ok
echo.
echo   [PROD MODE]
echo.
echo   All at  : http://localhost:8080
echo   API Docs: http://localhost:8080/docs
echo   Health  : http://localhost:8080/health
echo.
echo   Press Ctrl+C to stop
echo =========================================
echo.

:start
%PYTHON_CMD% -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload --reload-dir=api --reload-dir=agent_by_langgraph --reload-dir=agent_core --no-access-log

pause
