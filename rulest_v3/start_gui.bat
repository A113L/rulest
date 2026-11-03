@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.9+ and try again.
  pause
  exit /b 1
)
if "%RULEST_HOST%"=="" set RULEST_HOST=0.0.0.0
python app.py --host "%RULEST_HOST%" --port 8765
pause
