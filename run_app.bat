@echo off
title 🌲 Forest Phenology v6 — Launcher
color 0A

echo.
echo  =========================================================
echo   🌲  Universal Indian Forest Phenology Predictor v6
echo  =========================================================
echo.

REM Check Python
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo  [ERROR] Python not found. Please install Python 3.9+ from:
    echo          https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo  [1/3] Checking/installing dependencies...
pip install streamlit pandas numpy scipy scikit-learn matplotlib --quiet --disable-pip-version-check
IF ERRORLEVEL 1 (
    echo  [ERROR] Failed to install packages. Check your internet connection.
    pause
    exit /b 1
)

echo  [2/3] Dependencies ready.
echo  [3/3] Starting app — browser will open automatically...
echo.
echo  Press Ctrl+C in this window to stop the app.
echo.

streamlit run app/universal_Indian_forest_phenology_v6.py --server.headless false --browser.gatherUsageStats false

pause
