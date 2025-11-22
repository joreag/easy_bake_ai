#!/bin/bash

echo "==============================================================================="
echo " HCTS Easy Bake AI - Forge V1"
echo "==============================================================================="

# 1. Check for Python
if ! command -v python3 &> /dev/null
then
    echo "[ERROR] Python3 could not be found."
    echo "Please install Python3 to continue."
    exit
fi

# 2. Create Virtual Env
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv .venv
fi

# 3. Activate and Install
source .venv/bin/activate

echo "[INFO] Checking dependencies..."
pip install -r requirements.txt > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "[WARNING] Installing dependencies (First Run)..."
    pip install -r requirements.txt
fi

# 4. Launch
echo ""
echo "[SUCCESS] Launching Easy Bake AI..."
echo "Open http://127.0.0.1:5555 in your browser."
echo ""

python3 main.py