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
REM 榛樿寮€鍙戞ā寮忥紙鍓嶅悗绔悓鏃跺惎鍔?+ 鎵撳紑娴忚鍣級
REM 璁?PROD_MODE=true 鍒欏彧鍚姩鍚庣锛堢敓浜фā寮忥級
if "%PROD_MODE%"=="1"   set PROD_MODE=true
if "%PROD_MODE%"=="yes" set PROD_MODE=true
if /i "%PROD_MODE%"=="true" (goto :prod_mode) else (goto :dev_mode)

:dev_mode
echo.
echo   [DEV MODE] 姝ｅ湪鍚姩鍓嶅悗绔湇鍔?..
echo.

REM -- 纭繚 npm 渚濊禆宸插畨瑁?------------------------------------
if not exist "frontend\node_modules" (
    echo [..] npm install...
    cd frontend
    call npm install 2>nul
    cd ..
)

REM -- 鍚姩 Vite 鍓嶇锛堟柊绐楀彛锛?----------------------------------
echo [..] 鍚姩鍓嶇 Vite (http://localhost:5173)...
start "SmartCS Frontend" cmd /c "cd /d %~dp0frontend && npm run dev"

REM -- 绛夊緟涓€涓嬭 Vite 鍏堣捣鏉?------------------------------------
timeout /t 3 /nobreak >nul

REM -- 鍚姩鍚庣锛堟柊绐楀彛锛?----------------------------------------
echo [..] 鍚姩鍚庣 API (http://localhost:8080)...
start "SmartCS Backend" cmd /c "cd /d %~dp0 && %PYTHON_CMD% -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload --reload-dir=api --reload-dir=agent_by_langgraph --reload-dir=agent_core --no-access-log"

REM -- 鎵撳紑娴忚鍣?-------------------------------------------------
timeout /t 2 /nobreak >nul
echo [OK] 姝ｅ湪鎵撳紑娴忚鍣?..
start http://localhost:5173

echo.
echo =========================================
echo   鏈嶅姟姝ｅ湪鍚姩锛岃绋嶅€?..
echo.
echo   鍓嶇椤甸潰 : http://localhost:5173
echo   鍚庣 API  : http://localhost:8080
echo   API 鏂囨。  : http://localhost:8080/docs
echo.
echo   鍏抽棴鍓嶇/鍚庣绐楀彛鍗冲彲鍋滄鏈嶅姟銆?echo =========================================
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
