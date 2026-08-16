@echo off
cd /d "%~dp0"
python launch_dashboard.py
if errorlevel 1 (
  echo.
  echo Dashboard startup failed. Review the message above.
  pause
)
