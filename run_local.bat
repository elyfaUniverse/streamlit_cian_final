@echo off
setlocal
cd /d "%~dp0"
set "STUDY_PYTHON=%USERPROFILE%\.venvs\study-py312\Scripts\python.exe"

if exist "%STUDY_PYTHON%" goto run
set "STUDY_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%STUDY_PYTHON%" goto install
python -m venv .venv
if errorlevel 1 exit /b 1

:install
"%STUDY_PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

:run
"%STUDY_PYTHON%" check_bundle.py
if errorlevel 1 exit /b 1
"%STUDY_PYTHON%" -m streamlit run app.py %*
exit /b %errorlevel%
