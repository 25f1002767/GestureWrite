@echo off
title AI Air Writing (Virtual Air Canvas)
cd /d "%~dp0"
py -3.13 main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [Error] Failed to start Air Writing.
    pause
)
