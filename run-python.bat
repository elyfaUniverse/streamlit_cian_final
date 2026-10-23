@echo off
setlocal
cd /d "%~dp0"
set "STUDY_PYTHON=%USERPROFILE%\anaconda3\python.exe"
if exist "%STUDY_PYTHON%" goto run
set "STUDY_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%STUDY_PYTHON%" goto run
echo Run run_local.bat to prepare the project Python environment.
exit /b 1

:run
"%STUDY_PYTHON%" %*
exit /b %errorlevel%
