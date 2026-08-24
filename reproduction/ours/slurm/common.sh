#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)
VENV_DIR=${PROJECT_DIR}/.venv
CACHE_ROOT=${RGCLS_CACHE_ROOT:-${PROJECT_DIR}/.cache/huggingface}
export HF_HOME=${HF_HOME:-${CACHE_ROOT}}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${CACHE_ROOT}/datasets}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${CACHE_ROOT}/hub}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

if command -v module >/dev/null 2>&1; then
  module load miniconda3/24.11.1
fi
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"
