@echo off
REM Greenhouse Job Application Agent - Windows Run Script
REM
REM Usage:
REM   run.bat              - Run in continuous mode
REM   run.bat --once       - Run once and exit
REM   run.bat --check      - Check configuration
REM

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================
echo   Greenhouse Job Application Agent
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [OK] Python found

REM Check if .env exists
if not exist ".env" (
    if exist ".env.example" (
        echo [!] .env not found, copying from .env.example
        copy .env.example .env >nul
        echo [!] Please edit .env with your API keys
        echo.
        echo     notepad .env
        echo.
        pause
        exit /b 1
    )
)

REM Load .env file
if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
        set "line=%%a"
        if not "!line:~0,1!"=="#" (
            if not "%%a"=="" (
                set "%%a=%%b"
            )
        )
    )
    echo [OK] Loaded .env configuration
)

REM Check required variables
if "%OPENAI_API_KEY%"=="" (
    echo [ERROR] OPENAI_API_KEY not set in .env
    echo         Edit .env and add your OpenAI API key
    pause
    exit /b 1
)
if "%OPENAI_API_KEY%"=="sk-your-openai-api-key-here" (
    echo [ERROR] OPENAI_API_KEY not configured
    echo         Edit .env and add your real OpenAI API key
    pause
    exit /b 1
)
echo [OK] OPENAI_API_KEY configured

if "%GH_BOARD_TOKEN%"=="" (
    echo [ERROR] GH_BOARD_TOKEN not set in .env
    pause
    exit /b 1
)
if "%GH_BOARD_TOKEN%"=="examplecompany" (
    echo [ERROR] GH_BOARD_TOKEN not configured
    echo         Edit .env and set a real board token (e.g., anthropic, stripe)
    pause
    exit /b 1
)
echo [OK] GH_BOARD_TOKEN: %GH_BOARD_TOKEN%

REM Check resume
if exist "resume.md" (
    echo [OK] Resume found: resume.md
) else (
    echo [!] No resume.md found - will use default
)

echo.
echo Configuration:
echo   Name:      %USER_FIRST_NAME% %USER_LAST_NAME%
echo   Email:     %USER_EMAIL%
echo   Job Query: %JOB_QUERY%
echo   Locations: %JOB_LOCATIONS%
echo.

REM Handle arguments
if "%1"=="--check" (
    echo Configuration check complete.
    pause
    exit /b 0
)

if "%1"=="--once" (
    echo Running once...
    set RUN_ONCE=true
    python agent.py
    pause
    exit /b 0
)

if "%1"=="--help" (
    echo Usage:
    echo   run.bat           Run in continuous mode
    echo   run.bat --once    Run once and exit
    echo   run.bat --check   Check configuration only
    echo   run.bat --help    Show this help
    pause
    exit /b 0
)

REM Default: continuous mode
echo Running in continuous mode (Ctrl+C to stop)...
echo.
python agent.py
pause
