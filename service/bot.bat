@echo off
:: ================================================================
:: Market Bot v26 — Service Management
:: Usage: bot [start|stop|restart|status|logs|tail|edit]
:: ================================================================

set NSSM=C:\tools\nssm\nssm.exe
set SVC=MarketBot
set BOT_DIR=C:\Users\mattg\Downloads\Claude

if "%1"=="start" (
    echo Starting %SVC%...
    %NSSM% start %SVC%
    goto :eof
)

if "%1"=="stop" (
    echo Stopping %SVC% (graceful, up to 30s)...
    %NSSM% stop %SVC%
    goto :eof
)

if "%1"=="restart" (
    echo Restarting %SVC%...
    %NSSM% restart %SVC%
    goto :eof
)

if "%1"=="status" (
    %NSSM% status %SVC%
    goto :eof
)

if "%1"=="logs" (
    goto :logs
)

if "%1"=="tail" (
    goto :tail
)

if "%1"=="edit" (
    echo Opening NSSM configuration GUI...
    %NSSM% edit %SVC%
    goto :eof
)

echo.
echo Market Bot v26 — Service Management
echo ====================================
echo.
echo Usage: bot [command]
echo.
echo Commands:
echo   start    Start the bot service
echo   stop     Graceful stop (sends Ctrl+C, waits 30s)
echo   restart  Stop then start
echo   status   Show if running/stopped
echo   logs     Print today's log file
echo   tail     Live-stream today's log (Ctrl+C to stop)
echo   edit     Open NSSM config GUI
echo.
goto :eof

:logs
:: Find today's log file
for /f "tokens=*" %%d in ('powershell -Command "Get-Date -Format 'yyyyMMdd'"') do set TODAY=%%d
set LOGFILE=%BOT_DIR%\logs\bot_%TODAY%.log
if exist "%LOGFILE%" (
    type "%LOGFILE%" | more
) else (
    echo No log file found for today: %LOGFILE%
)
goto :eof

:tail
:: Live tail of today's log
for /f "tokens=*" %%d in ('powershell -Command "Get-Date -Format 'yyyyMMdd'"') do set TODAY=%%d
set LOGFILE=%BOT_DIR%\logs\bot_%TODAY%.log
if exist "%LOGFILE%" (
    echo Tailing %LOGFILE% (Ctrl+C to stop)...
    echo.
    powershell -Command "Get-Content '%LOGFILE%' -Tail 50 -Wait"
) else (
    echo No log file found for today: %LOGFILE%
)
goto :eof
