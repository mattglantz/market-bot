@echo off
setlocal EnableDelayedExpansion
title Market Bot v25.3
color 0A

echo.
echo  ========================================================
echo    MARKET BOT v25.3 - ES Futures Trading Assistant
echo    Confidence Pipeline Fix Applied
echo  ========================================================
echo.

:: ─── Verify Python ───
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo  [X] Python not found. Install 3.10+ from https://python.org
    pause & exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do echo  [OK] Python %%v

:: ─── Verify bot file ───
if not exist "market_bot_v25_patched.py" (
    echo  [X] market_bot_v25_patched.py not found in %CD%
    pause & exit /b 1
)
echo  [OK] Bot file found

:: ─── Create .env from template if missing ───
if not exist ".env" (
    if exist ".env.template" (
        copy ".env.template" ".env" >nul
        echo  [!!] Created .env from template - edit it with your API keys first
        echo.
        notepad .env
        pause & exit /b 0
    ) else (
        echo  [X] No .env file. Create one with at minimum:
        echo       ANTHROPIC_API_KEY=sk-ant-...
        pause & exit /b 1
    )
)
echo  [OK] .env loaded

:: ─── Quick dependency check ───
set NEED_INSTALL=0
for %%p in (requests pandas numpy PIL anthropic yfinance dotenv win32gui) do (
    python -c "import %%p" 2>nul || set NEED_INSTALL=1
)
if !NEED_INSTALL! equ 1 (
    echo.
    echo  Installing dependencies...
    pip install requests pandas numpy Pillow anthropic yfinance python-dotenv pywin32 urllib3
    if !ERRORLEVEL! neq 0 (
        echo  [X] pip install failed
        pause & exit /b 1
    )
    echo  [OK] Dependencies installed
) else (
    echo  [OK] Dependencies present
)

:: ─── Verify companion modules ───
set MOD_WARN=0
for %%m in (fractal_engine.py ibkr_client.py advanced_features.py) do (
    if not exist "%%m" (
        echo  [!] Missing %%m
        set MOD_WARN=1
    )
)
if !MOD_WARN! equ 0 echo  [OK] Local modules found

:: ─── Create folders ───
if not exist "logs" mkdir logs
if not exist "chart_library" mkdir chart_library

:: ─── Show IBKR mode ───
set IBKR_MODE=TWS Paper (7497)
for /f "usebackq tokens=2 delims==" %%a in (`findstr /B "IBKR_PORT" .env 2^>nul`) do (
    if "%%a"=="7496" set IBKR_MODE=TWS LIVE (7496)
    if "%%a"=="4001" set IBKR_MODE=Gateway LIVE (4001)
    if "%%a"=="4002" set IBKR_MODE=Gateway Paper (4002)
)

echo.
echo  ------------------------------------------------
echo   IBKR:          !IBKR_MODE!
echo   Max Penalty:   -25 (was -70)
echo   Confluence:    Floor 58-72%% on 3+ signals
echo   Ctrl+C to stop
echo  ------------------------------------------------
echo.

timeout /t 2 /nobreak >nul
python market_bot_v26.py

echo.
echo  Bot exited (code %ERRORLEVEL%).
if %ERRORLEVEL% neq 0 echo  Check logs\ for details.
pause
