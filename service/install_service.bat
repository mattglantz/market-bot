@echo off
:: ================================================================
:: Install Market Bot v26 as a Windows Service via NSSM
:: Run this script as Administrator (right-click > Run as admin)
:: ================================================================

set NSSM=C:\tools\nssm\nssm.exe
set SVC=MarketBot
set PYTHON=C:\Python314\python.exe
set BOT_DIR=C:\Users\mattg\Downloads\Claude
set BOT_SCRIPT=market_bot_v26.py

:: Check NSSM exists
if not exist "%NSSM%" (
    echo ERROR: NSSM not found at %NSSM%
    echo.
    echo Download NSSM 2.24 from https://nssm.cc/download
    echo Extract win64\nssm.exe to C:\tools\nssm\nssm.exe
    echo.
    pause
    exit /b 1
)

:: Check Python exists
if not exist "%PYTHON%" (
    echo ERROR: Python not found at %PYTHON%
    pause
    exit /b 1
)

echo Installing service "%SVC%"...
echo.

:: Install the service
%NSSM% install %SVC% %PYTHON%
%NSSM% set %SVC% AppParameters "%BOT_SCRIPT% --headless"
%NSSM% set %SVC% AppDirectory %BOT_DIR%

:: Display name and description
%NSSM% set %SVC% DisplayName "Market Bot v26 - ES Trading"
%NSSM% set %SVC% Description "ES Futures Trading Bot (IBKR + Claude AI). Auto-restarts on crash."

:: Startup: Automatic (Delayed Start) -- gives IBKR TWS time to launch first
%NSSM% set %SVC% Start SERVICE_DELAYED_AUTO_START

:: Restart policy: restart on crash, 30s delay, throttle at 60s
%NSSM% set %SVC% AppExit Default Restart
%NSSM% set %SVC% AppRestartDelay 30000
%NSSM% set %SVC% AppThrottle 60000

:: Graceful shutdown: send Ctrl+C, wait 30 seconds for cleanup
:: (Bot disconnects IBKR, sends Telegram "OFFLINE" alert, stops threads)
%NSSM% set %SVC% AppStopMethodSkip 0
%NSSM% set %SVC% AppStopMethodConsole 30000
%NSSM% set %SVC% AppStopMethodWindow 0
%NSSM% set %SVC% AppStopMethodThreads 0

:: Stdout/stderr capture (supplement to bot's own log files)
%NSSM% set %SVC% AppStdout %BOT_DIR%\logs\service_stdout.log
%NSSM% set %SVC% AppStderr %BOT_DIR%\logs\service_stderr.log
%NSSM% set %SVC% AppStdoutCreationDisposition 4
%NSSM% set %SVC% AppStderrCreationDisposition 4
%NSSM% set %SVC% AppRotateFiles 1
%NSSM% set %SVC% AppRotateOnline 1
%NSSM% set %SVC% AppRotateBytes 10485760

:: Environment: ensure unbuffered output for real-time log capture
%NSSM% set %SVC% AppEnvironmentExtra "PYTHONUNBUFFERED=1"

echo.
echo ================================================================
echo Service "%SVC%" installed successfully!
echo ================================================================
echo.
echo IMPORTANT NEXT STEPS:
echo.
echo 1. Open services.msc (Win+R, type "services.msc")
echo 2. Find "Market Bot v26 - ES Trading"
echo 3. Right-click ^> Properties ^> Log On tab
echo 4. Select "This account" and enter your Windows credentials
echo    (needed for .env access, IBKR TWS, chart screenshots)
echo.
echo 5. Then start the service:
echo    service\bot.bat start
echo.
pause
