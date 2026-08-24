#!/usr/bin/env bash
#SBATCH --job-name=ic27_bootstrap
#SBATCH --account=mst114566
#SBATCH --partition=dev
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${PROJECT_DIR}"
mkdir -p logs
if command -v module >/dev/null 2>&1; then
  module load miniconda3/24.11.1
fi
if [[ ! -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
  python -m venv "${PROJECT_DIR}/.venv"
fi
source "${PROJECT_DIR}/.venv/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "${PROJECT_DIR}"
python -m pip check
