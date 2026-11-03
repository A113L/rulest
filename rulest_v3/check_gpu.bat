@echo off
setlocal
cd /d "%~dp0"
python gpu_diagnostics.py
echo.
echo Diagnostic exit code: %errorlevel%
pause
