#!/bin/bash
# =========================================================
#  🌲  Universal Indian Forest Phenology Predictor v6
#  One-click launcher for macOS / Linux
# =========================================================

cd "$(dirname "$0")"

echo ""
echo " ========================================================="
echo "  🌲  Universal Indian Forest Phenology Predictor v6"
echo " ========================================================="
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo " [ERROR] python3 not found."
    echo "         macOS: brew install python"
    echo "         Ubuntu/Debian: sudo apt install python3 python3-pip"
    read -p " Press Enter to exit..."
    exit 1
fi

echo " [1/3] Python found: $(python3 --version)"
echo " [2/3] Checking/installing dependencies..."

python3 -m pip install streamlit pandas numpy scipy scikit-learn matplotlib \
    --quiet --disable-pip-version-check 2>&1

if [ $? -ne 0 ]; then
    echo " [ERROR] Package installation failed. Check your internet connection."
    read -p " Press Enter to exit..."
    exit 1
fi

echo " [3/3] Starting app — browser will open automatically..."
echo ""
echo " Press Ctrl+C to stop the app."
echo ""

python3 -m streamlit run app/universal_Indian_forest_phenology_v6.py \
    --server.headless false \
    --browser.gatherUsageStats false

read -p " App stopped. Press Enter to exit..."
