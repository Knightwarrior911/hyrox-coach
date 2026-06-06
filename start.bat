@echo off
title HYROX Coach
echo ========================================
echo   HYROX COACH
echo ========================================
echo.

set PYTHON=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
set PATH=%LOCALAPPDATA%\Programs\Python\Python311;%LOCALAPPDATA%\Programs\Python\Python311\Scripts;%PATH%

%PYTHON% --version
echo.
echo Starting server...
echo.

cd /d %~dp0
%PYTHON% scripts\server.py

pause
