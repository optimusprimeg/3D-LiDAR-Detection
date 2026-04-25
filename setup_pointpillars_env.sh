#!/usr/bin/env bash
set -euo pipefail

# One-command bootstrap for PointPillars environment.
# Usage examples:
#   ./setup_pointpillars_env.sh
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 ./setup_pointpillars_env.sh
#   VENV_DIR=/custom/path/venv ./setup_pointpillars_env.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PP_DIR="$ROOT_DIR/PointPillars"
VENV_DIR="${VENV_DIR:-$PP_DIR/venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"

if [[ ! -d "$PP_DIR" ]]; then
  echo "Error: PointPillars directory not found at $PP_DIR"
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is required but was not found in PATH"
  exit 1
fi

echo "[1/5] Creating virtual environment at $VENV_DIR"
python3 -m venv "$VENV_DIR"

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "[2/5] Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

if [[ -n "$TORCH_INDEX_URL" ]]; then
  echo "[3/5] Installing torch/torchvision/torchaudio from $TORCH_INDEX_URL"
  python -m pip install --index-url "$TORCH_INDEX_URL" torch torchvision torchaudio
else
  echo "[3/5] Skipping explicit torch install (requirements.txt will handle it)"
fi

echo "[4/5] Installing PointPillars requirements"
python -m pip install -r "$PP_DIR/requirements.txt"

echo "[5/5] Installing PointPillars package"
python -m pip install -e "$PP_DIR"

echo
echo "Environment setup complete."
echo "Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo
echo "Run demo with:"
echo "  cd $ROOT_DIR"
echo "  ./PointPillars/run_demo_inference.py"
echo
echo "Run GPU demo (after CUDA setup):"
echo "  cd $ROOT_DIR"
echo "  python PointPillars/gpu_demo/run_gpu_inference.py"
