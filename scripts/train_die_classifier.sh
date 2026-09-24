#!/usr/bin/env bash
# Run train_die_classifier.py with the container venv, no activation needed.
# All arguments are forwarded, e.g.:
#   ./train_die_classifier.sh --device cuda --epochs 12 --wandb
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SCRIPT_DIR}/../.venv-docker/bin/python3"

if [ ! -x "${PY}" ]; then
    echo "Container venv not found. Create it first with:" >&2
    echo "  $(cd "${SCRIPT_DIR}/.." && pwd)/setup_venv_docker.sh" >&2
    exit 1
fi

exec "${PY}" "${SCRIPT_DIR}/train_die_classifier.py" "$@"
