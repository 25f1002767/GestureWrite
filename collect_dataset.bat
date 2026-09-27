@echo off
title AirWrite Dataset Studio
cd /d "%~dp0"
py -3.13 collect_dataset.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [Error] Failed to start Dataset Studio.
    pause
)
