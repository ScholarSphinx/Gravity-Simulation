@echo off
setlocal
cd /d "%~dp0"

set "EXE=build\Release\GravitySimulation.exe"

if exist "%EXE%" (
    echo Running %EXE% ...
    "%EXE%"
    goto :eof
)

echo Release executable not found. Building first...
powershell -NoProfile -ExecutionPolicy Bypass -File ".\build.ps1" -Config Release
if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

if not exist "%EXE%" (
    echo Executable still not found after build: %EXE%
    exit /b 1
)

echo Running %EXE% ...
"%EXE%"
