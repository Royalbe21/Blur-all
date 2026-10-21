@echo off
setlocal
cd /d "%~dp0\.."
where py >nul 2>nul
if errorlevel 1 (
  echo Install Python 3.11 or 3.12 from python.org first, including the Python launcher.
  pause
  exit /b 1
)
if not exist .venv\Scripts\python.exe (
  py -3.12 -m venv .venv
  if errorlevel 1 py -3.11 -m venv .venv
  if errorlevel 1 goto :failed
)
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :failed
.venv\Scripts\python.exe -m blur_all
if errorlevel 1 goto :failed
exit /b 0
:failed
echo Blur All could not start. Review the error above.
pause
exit /b 1
