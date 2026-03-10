@echo off
:: ================================================================
:: Uninstall Market Bot Windows Service
:: Run as Administrator
:: ================================================================

set NSSM=C:\tools\nssm\nssm.exe
set SVC=MarketBot

echo Stopping service if running...
%NSSM% stop %SVC% 2>nul

echo Removing service "%SVC%"...
%NSSM% remove %SVC% confirm

echo.
echo Service removed. Bot can still be run manually:
echo   cd C:\Users\mattg\Downloads\Claude
echo   python market_bot_v26.py
echo.
pause
