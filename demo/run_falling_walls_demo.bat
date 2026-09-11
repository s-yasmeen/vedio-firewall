@echo off
setlocal
cd /d %~dp0\..

where py >nul 2>nul
if %errorlevel%==0 (
  set PY=py
) else (
  set PY=python
)

%PY% -m pip install -r demo\requirements-falling-walls.txt
if errorlevel 1 goto :fail

%PY% -m streamlit run demo\falling_walls_demo.py --server.headless true
exit /b 0

:fail
echo.
echo TAPF-MIN demo setup failed. Check that Python 3.11+ and internet access are available for the first installation.
pause
exit /b 1
