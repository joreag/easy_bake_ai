@echo off
setlocal

TITLE Easy Bake AI Launcher

echo ===============================================================================
echo  HCTS Easy Bake AI - Forge V1
echo ===============================================================================

:: 1. Check if Python is installed
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please download Python 3.10 or 3.11 from python.org.
    echo IMPORTANT: Check the box "Add Python to PATH" during installation.
    pause
    exit /b
)

:: 2. Create Virtual Environment if it doesn't exist
IF NOT EXIST ".venv" (
    echo [INFO] Creating virtual environment... this happens only once.
    python -m venv .venv
    echo [INFO] Virtual environment created.
)

:: 3. Activate Environment
call .venv\Scripts\activate.bat

:: 4. Install Dependencies
echo [INFO] Checking dependencies...
pip install -r requirements.txt >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Installing dependencies for the first time... please wait.
    echo This might take a few minutes depending on your internet speed.
    pip install -r requirements.txt
)

:: 5. Launch the App
echo.
echo [SUCCESS] System ready.
echo [INFO] Launching Inference Engine...
echo.
echo Open your browser to: http://127.0.0.1:5555
echo (Keep this black window open while using the AI)
echo.

python main.py

pause