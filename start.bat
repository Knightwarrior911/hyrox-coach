@echo off
title HYROX Coach
echo ========================================
echo   HYROX COACH
echo ========================================
echo.
echo Starting server...
echo.

:: Set paths
set PYTHON=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
set PATH=%LOCALAPPDATA%\Programs\Python\Python311;%LOCALAPPDATA%\Programs\Python\Python311\Scripts;%PATH%

:: Kill any existing python processes first (avoid port conflicts)
taskkill /IM python.exe /F 2>nul

:: Wait a moment for port to free up
timeout /t 2 /nobreak >nul

:: Check Python exists
if not exist "%PYTHON%" (
    echo ERROR: Python not found at %PYTHON%
    echo Please install Python 3.11 or update the path in this shortcut.
    pause
    exit /b 1
)

:: Go to this script's directory
cd /d %~dp0

:: Install/update dependencies silently
echo Checking dependencies...
%PYTHON% -m pip install -r requirements.txt --quiet 2>nul

:: Start server
echo.
echo Launching HYROX Coach server...
echo.
%PYTHON% -u scripts\server.py

:: If server exits, pause so user can see error
echo.
echo Server stopped. Press any key to close...
pause >nul
