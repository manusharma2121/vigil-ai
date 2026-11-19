@echo off
REM ── VIGIL AI – Start Backend ──────────────────────────────────────────────────
echo.
echo  ██╗   ██╗██╗ ██████╗ ██╗██╗      █████╗ ██╗
echo  ██║   ██║██║██╔════╝ ██║██║     ██╔══██╗██║
echo  ██║   ██║██║██║  ███╗██║██║     ███████║██║
echo  ╚██╗ ██╔╝██║██║   ██║██║██║     ██╔══██║██║
echo   ╚████╔╝ ██║╚██████╔╝██║███████╗██║  ██║██║
echo    ╚═══╝  ╚═╝ ╚═════╝ ╚═╝╚══════╝╚═╝  ╚═╝╚═╝
echo.
echo  Intelligent Video Violence Detection System
echo  ─────────────────────────────────────────────
echo.

cd /d "%~dp0backend"

REM Check for .env file
if not exist ".env" (
    echo  [!] No .env file found. Copying from .env.example...
    copy "..\\.env.example" ".env" >nul 2>&1
)

REM Create required directories
if not exist "..\models\weights" mkdir "..\models\weights"
if not exist "..\uploads" mkdir "..\uploads"
if not exist "..\datasets\annotations" mkdir "..\datasets\annotations"

echo  [*] Starting FastAPI backend on http://localhost:8000
echo  [*] API docs available at http://localhost:8000/docs
echo  [*] Press Ctrl+C to stop
echo.

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause
