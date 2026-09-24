#!/usr/bin/env bash
# Training / YOLOE venv for the ROS 2 Humble docker container (Python 3.10).
#
# Unlike .venv/ (created on the host, whose Python version differs from the
# container's), this venv lives on the bind-mounted workspace so it survives
# container restarts, and uses --system-site-packages so rclpy, cv_bridge and
# the rest of /opt/ros/humble stay importable after activation.
#
#   ./setup_venv_docker.sh
#   source .venv-docker/bin/activate
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv-docker"
# RTX 50xx (Blackwell, sm_120) needs CUDA >= 12.8 wheels
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

echo "=== Creating Python virtual environment in ${VENV_DIR} ==="
if [ ! -x "${VENV_DIR}/bin/python3" ]; then
    # Ubuntu's python3 lacks ensurepip unless python3-venv is installed
    if ! python3 -m venv --system-site-packages "${VENV_DIR}" 2>/dev/null; then
        rm -rf "${VENV_DIR}"
        python3 -m venv --system-site-packages --without-pip "${VENV_DIR}"
        curl -sS https://bootstrap.pypa.io/get-pip.py | "${VENV_DIR}/bin/python3"
    fi
fi

echo "=== Installing PyTorch from ${TORCH_INDEX} ==="
"${VENV_DIR}/bin/pip" install torch torchvision --index-url "${TORCH_INDEX}"

echo "=== Installing detection (requirements.txt) + training dependencies ==="
# requirements.txt pins numpy<2, which ROS 2 Humble's cv_bridge also needs;
# torch is already satisfied by the CUDA build above, so pip keeps it.
# Ultralytics' CLIP fork provides the tokenizer of YOLOE's MobileCLIP text encoder
# (set_classes). It is not a pip dependency of ultralytics, which otherwise installs
# it on the node's first launch and fails that launch ("No module named 'clip'").
"${VENV_DIR}/bin/pip" install -r "${SCRIPT_DIR}/requirements.txt" ultralytics \
    "git+https://github.com/ultralytics/CLIP.git" wandb tqdm pyrender trimesh

"${VENV_DIR}/bin/python3" -c "import torch; print(f'torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
"${VENV_DIR}/bin/python3" -c "import ultralytics, clip; print(f'ultralytics {ultralytics.__version__}, clip OK')"

echo "=== Setup complete! ==="
echo "Activate with:  source ${VENV_DIR}/bin/activate"
echo "Or train with:  ${SCRIPT_DIR}/scripts/train_die_classifier.sh [--wandb ...]"
