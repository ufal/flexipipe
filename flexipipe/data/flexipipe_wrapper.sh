#!/bin/bash
# Wrapper script for flexipipe
# This allows running 'flexipipe' directly instead of 'python -m flexipipe'
#
# This script works with:
# - pip-installed flexipipe (system-wide or user)
# - Development installations (cloned repo)
# - Virtual environments (automatic detection or VENV_PATH)
#
# Configuration:
#   Set FLEXIPIPE_REPO_PATH to point to a cloned flexipipe repository
#   Set VENV_PATH to use a specific virtual environment

# Optional: Set path to flexipipe repository (for development installations)
# Uncomment and modify if using a cloned repo:
# FLEXIPIPE_REPO_PATH="/path/to/flexipipe"

# Optional: Set a specific virtual environment path
# Uncomment and modify if you want to use a specific venv:
# VENV_PATH="/path/to/venv"
# Example for your setup:
# VENV_PATH="/Volumes/Data2/Flexipipe/venv"
# FLEXIPIPE_REPO_PATH="/Users/mjanssen/programming/flexipipe"

# Function to find Python executable
find_python() {
    # If VENV_PATH is set, use it
    if [ -n "$VENV_PATH" ] && [ -f "$VENV_PATH/bin/python" ]; then
        echo "$VENV_PATH/bin/python"
        return 0
    fi
    
    # Check if we're in a virtual environment
    if [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/bin/python" ]; then
        echo "$VIRTUAL_ENV/bin/python"
        return 0
    fi
    
    # Check for common Python executables
    for py in python3 python; do
        if command -v "$py" >/dev/null 2>&1; then
            echo "$py"
            return 0
        fi
    done
    
    return 1
}

# Function to check if flexipipe module is available (lightweight: no full import)
check_flexipipe() {
    local python_exe="$1"
    "$python_exe" -c "import importlib.util; exit(0 if importlib.util.find_spec('flexipipe') else 1)" 2>/dev/null
}

# Function to check if we're in a flexipipe repo directory
is_flexipipe_repo() {
    local path="$1"
    [ -f "$path/flexipipe/__main__.py" ] && [ -f "$path/setup.py" ]
}

# Find Python executable
PYTHON_EXE=$(find_python)
if [ $? -ne 0 ]; then
    echo "Error: Python not found. Please install Python 3." >&2
    exit 1
fi

# First, check if flexipipe is available in the current Python environment
# (This takes priority - if it's installed via pip, use that)
if check_flexipipe "$PYTHON_EXE"; then
    # Installed via pip - use it directly
    exec "$PYTHON_EXE" -m flexipipe "$@"
fi

# If not found, check for development installation
USE_DEV_INSTALL=false
DEV_REPO_PATH=""

# If FLEXIPIPE_REPO_PATH is set, use it
if [ -n "$FLEXIPIPE_REPO_PATH" ]; then
    if is_flexipipe_repo "$FLEXIPIPE_REPO_PATH"; then
        USE_DEV_INSTALL=true
        DEV_REPO_PATH="$FLEXIPIPE_REPO_PATH"
    else
        echo "Warning: FLEXIPIPE_REPO_PATH is set but doesn't appear to be a flexipipe repository: $FLEXIPIPE_REPO_PATH" >&2
    fi
fi

# If not set, try to find flexipipe repo in common locations
if [ "$USE_DEV_INSTALL" = false ]; then
    # Check current directory and parent directories (up to 3 levels)
    current_dir="$(pwd)"
    for i in 0 1 2 3; do
        check_path="$current_dir"
        for ((j=0; j<i; j++)); do
            check_path="$(dirname "$check_path")"
        done
        if is_flexipipe_repo "$check_path"; then
            USE_DEV_INSTALL=true
            DEV_REPO_PATH="$check_path"
            break
        fi
    done
fi

# Run flexipipe
if [ "$USE_DEV_INSTALL" = true ]; then
    # Development installation: add repo to PYTHONPATH and run
    # Make sure we use absolute path
    DEV_REPO_PATH="$(cd "$DEV_REPO_PATH" && pwd)"
    export PYTHONPATH="$DEV_REPO_PATH${PYTHONPATH:+:$PYTHONPATH}"
    # Verify it works before executing
    if ! "$PYTHON_EXE" -c "import flexipipe" 2>/dev/null; then
        echo "Error: Failed to import flexipipe from development installation at: $DEV_REPO_PATH" >&2
        echo "Make sure the flexipipe package is properly set up in that directory." >&2
        exit 1
    fi
    exec "$PYTHON_EXE" -m flexipipe "$@"
else
    echo "Error: flexipipe module not found." >&2
    echo "" >&2
    echo "Python executable: $PYTHON_EXE" >&2
    echo "" >&2
    echo "Options:" >&2
    echo "  1. Install via pip: pip install flexipipe" >&2
    echo "  2. Use development installation:" >&2
    echo "     - Set FLEXIPIPE_REPO_PATH=/path/to/flexipipe" >&2
    echo "     - Or run from within a flexipipe repository directory" >&2
    if [ -n "$VENV_PATH" ]; then
        echo "  3. Activate your virtual environment: source $VENV_PATH/bin/activate" >&2
    fi
    exit 1
fi
