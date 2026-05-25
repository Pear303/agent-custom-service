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
REM 默认开发模式（前后端同时启动 + 打开浏览器）
REM 设 PROD_MODE=true 则只启动后端（生产模式）
if "%PROD_MODE%"=="1"   set PROD_MODE=true
if "%PROD_MODE%"=="yes" set PROD_MODE=true
if /i "%PROD_MODE%"=="true" (goto :prod_mode) else (goto :dev_mode)

:dev_mode
echo.
echo   [DEV MODE] 正在启动前后端服务...
echo.

REM -- 确保 npm 依赖已安装 ------------------------------------
if not exist "frontend\node_modules" (
    echo [..] npm install...
    cd frontend
    call npm install 2>nul
    cd ..
)

REM -- 启动 Vite 前端（新窗口） ----------------------------------
echo [..] 启动前端 Vite (http://localhost:5173)...
start "SmartCS Frontend" cmd /c "cd /d %~dp0frontend && npm run dev"

REM -- 等待一下让 Vite 先起来 ------------------------------------
timeout /t 3 /nobreak >nul

REM -- 启动后端（新窗口） ----------------------------------------
echo [..] 启动后端 API (http://localhost:8080)...
start "SmartCS Backend" cmd /c "cd /d %~dp0 && %PYTHON_CMD% -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload --reload-dir=api --reload-dir=agent"

REM -- 打开浏览器 -------------------------------------------------
timeout /t 2 /nobreak >nul
echo [OK] 正在打开浏览器...
start http://localhost:5173

echo.
echo =========================================
echo   服务正在启动，请稍候...
echo.
echo   前端页面 : http://localhost:5173
echo   后端 API  : http://localhost:8080
echo   API 文档  : http://localhost:8080/docs
echo.
echo   关闭前端/后端窗口即可停止服务。
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
%PYTHON_CMD% -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload --reload-dir=api --reload-dir=agent

pause
