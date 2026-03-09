#!/usr/bin/env bash
# OpenMic Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/fushipanda/openmic/main/install.sh | bash
set -euo pipefail

REPO="fushipanda/openmic"
MIN_PYTHON="3.12"

echo ""
echo "  ██████  ██████  ███████ ███    ██ ███    ███ ██  ██████"
echo " ██    ██ ██   ██ ██      ████   ██ ████  ████ ██ ██"
echo " ██    ██ ██████  █████   ██ ██  ██ ██ ████ ██ ██ ██"
echo " ██    ██ ██      ██      ██  ██ ██ ██  ██  ██ ██ ██"
echo "  ██████  ██      ███████ ██   ████ ██      ██ ██  ██████"
echo ""
echo " Installer"
echo ""

# --- Check Python version ---
check_python() {
    local py=""
    for cmd in python3 python; do
        if command -v "$cmd" &>/dev/null; then
            local ver
            ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
            local major minor
            major=$(echo "$ver" | cut -d. -f1)
            minor=$(echo "$ver" | cut -d. -f2)
            if [ "$major" -ge 3 ] && [ "$minor" -ge 12 ]; then
                py="$cmd"
                break
            fi
        fi
    done
    if [ -z "$py" ]; then
        echo "Error: Python ${MIN_PYTHON}+ is required but not found."
        echo "Install Python from https://www.python.org/downloads/"
        exit 1
    fi
    echo "Found $py ($($py --version 2>&1))"
    PYTHON="$py"
}

# --- Install ---
install_openmic() {
    local pkg="git+https://github.com/${REPO}.git"

    # Try pipx first (isolated install)
    if command -v pipx &>/dev/null; then
        echo "Installing with pipx..."
        pipx install "$pkg"
        return
    fi

    # Try uv next
    if command -v uv &>/dev/null; then
        echo "Installing with uv..."
        uv tool install "$pkg"
        return
    fi

    # Fall back to pip
    echo "Installing with pip..."
    "$PYTHON" -m pip install --user "$pkg"
    echo ""
    echo "Note: Make sure ~/.local/bin is in your PATH."
}

# --- Main ---
check_python
install_openmic

echo ""
echo "Installation complete!"
echo ""
echo "Run 'openmic' to start (the setup wizard will guide you through configuration)."
echo ""
