@echo off
REM ── VIGIL AI – Start Frontend ─────────────────────────────────────────────────
echo.
echo  Starting VIGIL AI frontend dev server...
echo  URL: http://localhost:5173
echo  Press Ctrl+C to stop
echo.

cd /d "%~dp0frontend"
npx vite

pause
