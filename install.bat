@echo off
REM ── VIGIL AI – Install All Dependencies ──────────────────────────────────────
echo.
echo  Installing VIGIL AI dependencies...
echo.

echo  [1/2] Installing Python backend dependencies...
cd /d "%~dp0"
python -m pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo  [!] pip install failed. Make sure Python 3.9+ is installed and in PATH.
    pause
    exit /b 1
)

echo.
echo  [2/2] Installing Node.js frontend dependencies...
cd /d "%~dp0frontend"
npm install
if %ERRORLEVEL% NEQ 0 (
    echo  [!] npm install failed. Make sure Node.js is installed.
    pause
    exit /b 1
)

echo.
echo  ✓ All dependencies installed successfully!
echo.
echo  To start the application:
echo    1. Run start_backend.bat  (FastAPI on port 8000)
echo    2. Run start_frontend.bat (React on port 5173)
echo    3. Open http://localhost:5173 in your browser
echo.

pause
