#!/bin/bash

# Get the directory where this script sits
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VENV_PATH="$SCRIPT_DIR/venv"
PYTHON_BIN="$VENV_PATH/bin/python3"

if [ ! -f "$PYTHON_BIN" ]; then
    echo "❌ Error: Virtual environment not found at $VENV_PATH"
    echo "Please ensure you are in the project folder."
    exit 1
fi

echo "🚀 Starting Blood Group Detection with Hardware Acceleration..."

# Dynamically find all NVIDIA library paths in the venv's site-packages.
# This ensures TensorFlow can find cuDNN, cuBLAS, etc. without global installs.
export LD_LIBRARY_PATH=$($PYTHON_BIN -c "import os; import nvidia; base_path = os.path.dirname(nvidia.__file__); print(':'.join([os.path.join(base_path, lib, 'lib') for lib in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, lib, 'lib'))]))"):$LD_LIBRARY_PATH

# Run the app
$PYTHON_BIN "$SCRIPT_DIR/main.py" "$@"
