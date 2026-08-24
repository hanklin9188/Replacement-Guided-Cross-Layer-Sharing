#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${FAD_WORKSPACE_ROOT:-$(cd "${PROJECT_ROOT}/../.." && pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

DEFAULT_RUN_ROOT="${WORKSPACE_ROOT}/out/newthesis_final_llama_proto19_20pct_s44_20260621_174818"
RUN_ROOT="${RUN_ROOT:-${DEFAULT_RUN_ROOT}}"
DEPLOY_BUNDLE="${DEPLOY_BUNDLE:-${RUN_ROOT}/phase4_export/deploy_bundle.pt}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-${WORKSPACE_ROOT}/data/datasets}"
RUN_TAG="${RUN_TAG:-confidence_sweep_proto19_20pct_s44_$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${RESULTS_DIR:-${WORKSPACE_ROOT}/results/${RUN_TAG}}"

CONF_SWEEP_SCRIPT="${CONF_SWEEP_SCRIPT:-${PROJECT_ROOT}/core/confidence_threshold_sweep_exit_eval_final_llama.py}"
CONF_SWEEP_DATASETS="${CONF_SWEEP_DATASETS:-piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa}"
CONF_SWEEP_EXIT_LAYERS="${CONF_SWEEP_EXIT_LAYERS:-12,16,20,24}"
CONF_SWEEP_THRESHOLDS="${CONF_SWEEP_THRESHOLDS:-0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,0.995}"
CONF_SWEEP_MAX_SAMPLES="${CONF_SWEEP_MAX_SAMPLES:-0}"
CONF_SWEEP_LENGTH_NORM="${CONF_SWEEP_LENGTH_NORM:-none}"
CONF_SWEEP_TEMPERATURE="${CONF_SWEEP_TEMPERATURE:-1.0}"
CONF_SWEEP_SAVE_RECORDS="${CONF_SWEEP_SAVE_RECORDS:-False}"
CONF_SWEEP_USE_QUANT_BANK_INT4="${CONF_SWEEP_USE_QUANT_BANK_INT4:-False}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"

mkdir -p "${RESULTS_DIR}"

if [ ! -f "${CONF_SWEEP_SCRIPT}" ]; then
  echo "[ConfSweep][Error] script not found: ${CONF_SWEEP_SCRIPT}" >&2
  exit 1
fi
if [ ! -f "${DEPLOY_BUNDLE}" ]; then
  echo "[ConfSweep][Error] deploy bundle not found: ${DEPLOY_BUNDLE}" >&2
  exit 1
fi

echo "[ConfSweep] run_tag=${RUN_TAG}"
echo "[ConfSweep] deploy_bundle=${DEPLOY_BUNDLE}"
echo "[ConfSweep] results_dir=${RESULTS_DIR}"
echo "[ConfSweep] datasets=${CONF_SWEEP_DATASETS}"
echo "[ConfSweep] exit_layers=${CONF_SWEEP_EXIT_LAYERS}"
echo "[ConfSweep] thresholds=${CONF_SWEEP_THRESHOLDS}"

"${PYTHON_BIN}" "${CONF_SWEEP_SCRIPT}" \
  --deploy_bundle "${DEPLOY_BUNDLE}" \
  --test_data_root "${TEST_DATA_ROOT}" \
  --datasets "${CONF_SWEEP_DATASETS}" \
  --output_dir "${RESULTS_DIR}" \
  --max_samples "${CONF_SWEEP_MAX_SAMPLES}" \
  --exit_layers "${CONF_SWEEP_EXIT_LAYERS}" \
  --thresholds "${CONF_SWEEP_THRESHOLDS}" \
  --length_norm "${CONF_SWEEP_LENGTH_NORM}" \
  --temperature "${CONF_SWEEP_TEMPERATURE}" \
  --use_quant_bank_int4 "${CONF_SWEEP_USE_QUANT_BANK_INT4}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --trust_remote_code "${TRUST_REMOTE_CODE}" \
  --save_records "${CONF_SWEEP_SAVE_RECORDS}"

echo "[ConfSweep] done: ${RESULTS_DIR}"
