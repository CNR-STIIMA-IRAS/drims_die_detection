#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

echo "=== Creating Python virtual environment in ${VENV_DIR} ==="
if ! python3 -m venv "${VENV_DIR}" 2>/dev/null; then
    python3 -m venv --without-pip "${VENV_DIR}"
    curl -sS https://bootstrap.pypa.io/get-pip.py | "${VENV_DIR}/bin/python3"
fi

echo "=== Upgrading pip and installing dependencies from requirements.txt ==="
"${VENV_DIR}/bin/pip" install --upgrade pip
"${VENV_DIR}/bin/pip" install -r "${SCRIPT_DIR}/requirements.txt" || {
    echo "=== Notice: Batch requirements install failed. Retrying per package (open3d is optional) ==="
    while IFS= read -r pkg || [ -n "$pkg" ]; do
        [[ -z "$pkg" || "$pkg" =~ ^# ]] && continue
        "${VENV_DIR}/bin/pip" install "$pkg" || echo "  ⚠️ Optional package '$pkg' could not be installed on this Python version — skipping."
    done < "${SCRIPT_DIR}/requirements.txt"
}

echo "=== Setup complete! ==="
echo "Virtual environment is ready at: ${VENV_DIR}"
echo "To activate manually, run: source .venv/bin/activate"
