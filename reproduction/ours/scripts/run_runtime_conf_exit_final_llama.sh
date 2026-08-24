#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${FAD_WORKSPACE_ROOT:-$(cd "${PROJECT_ROOT}/../.." && pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

DEFAULT_RUN_ROOT="${WORKSPACE_ROOT}/out/newthesis_final_llama_fad_13p4_hidden_pca_seed42_4gpu_20260622_001508"
RUN_ROOT="${RUN_ROOT:-${DEFAULT_RUN_ROOT}}"
RUNTIME_EXIT_BASE_MODEL_NAME_OR_PATH="${RUNTIME_EXIT_BASE_MODEL_NAME_OR_PATH:-}"
if [ -n "${RUNTIME_EXIT_BASE_MODEL_NAME_OR_PATH}" ]; then
  DEPLOY_BUNDLE="${DEPLOY_BUNDLE:-}"
else
  DEPLOY_BUNDLE="${DEPLOY_BUNDLE:-${RUN_ROOT}/phase4_export/deploy_bundle.pt}"
fi
TEST_DATA_ROOT="${TEST_DATA_ROOT:-${WORKSPACE_ROOT}/data/datasets}"
RUN_TAG="${RUN_TAG:-runtime_conf_exit_08410_13p4_$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${RESULTS_DIR:-${WORKSPACE_ROOT}/results/${RUN_TAG}}"

RUNTIME_EXIT_SCRIPT="${RUNTIME_EXIT_SCRIPT:-${PROJECT_ROOT}/core/runtime_confidence_exit_eval_final_llama.py}"
RUNTIME_EXIT_DATASETS="${RUNTIME_EXIT_DATASETS:-piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa}"
RUNTIME_EXIT_LAYERS="${RUNTIME_EXIT_LAYERS:-12,16,20,24}"
RUNTIME_EXIT_THRESHOLD="${RUNTIME_EXIT_THRESHOLD:-0.90}"
RUNTIME_EXIT_DATASET_THRESHOLDS="${RUNTIME_EXIT_DATASET_THRESHOLDS:-}"
RUNTIME_EXIT_CONTROLLER_JSON="${RUNTIME_EXIT_CONTROLLER_JSON:-}"
RUNTIME_EXIT_CONTROLLER_DECISION_THRESHOLD="${RUNTIME_EXIT_CONTROLLER_DECISION_THRESHOLD:--1}"
RUNTIME_EXIT_MAX_SAMPLES="${RUNTIME_EXIT_MAX_SAMPLES:-0}"
RUNTIME_EXIT_LENGTH_NORM="${RUNTIME_EXIT_LENGTH_NORM:-none}"
RUNTIME_EXIT_TEMPERATURE="${RUNTIME_EXIT_TEMPERATURE:-1.0}"
RUNTIME_EXIT_ACTIVE_COMPACTION_BATCH_SIZE="${RUNTIME_EXIT_ACTIVE_COMPACTION_BATCH_SIZE:-1}"
RUNTIME_EXIT_RUN_FULL_BASELINE="${RUNTIME_EXIT_RUN_FULL_BASELINE:-True}"
RUNTIME_EXIT_SAVE_RECORDS="${RUNTIME_EXIT_SAVE_RECORDS:-False}"
RUNTIME_EXIT_WARMUP_SAMPLES="${RUNTIME_EXIT_WARMUP_SAMPLES:-0}"
RUNTIME_EXIT_USE_QUANT_BANK_INT4="${RUNTIME_EXIT_USE_QUANT_BANK_INT4:-False}"
RUNTIME_EXIT_WEIGHT_QUANTIZATION="${RUNTIME_EXIT_WEIGHT_QUANTIZATION:-none}"
RUNTIME_EXIT_QUANTIZE_LM_HEAD="${RUNTIME_EXIT_QUANTIZE_LM_HEAD:-False}"
RUNTIME_EXIT_SAVE_QUANTIZED_CHECKPOINT="${RUNTIME_EXIT_SAVE_QUANTIZED_CHECKPOINT:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"

mkdir -p "${RESULTS_DIR}"

if [ ! -f "${RUNTIME_EXIT_SCRIPT}" ]; then
  echo "[RuntimeExit][Error] script not found: ${RUNTIME_EXIT_SCRIPT}" >&2
  exit 1
fi
if [ -z "${DEPLOY_BUNDLE}" ] && [ -z "${RUNTIME_EXIT_BASE_MODEL_NAME_OR_PATH}" ]; then
  echo "[RuntimeExit][Error] set DEPLOY_BUNDLE or RUNTIME_EXIT_BASE_MODEL_NAME_OR_PATH" >&2
  exit 1
fi
if [ -n "${DEPLOY_BUNDLE}" ] && [ ! -f "${DEPLOY_BUNDLE}" ]; then
  echo "[RuntimeExit][Error] deploy bundle not found: ${DEPLOY_BUNDLE}" >&2
  exit 1
fi

echo "[RuntimeExit] run_tag=${RUN_TAG}"
echo "[RuntimeExit] deploy_bundle=${DEPLOY_BUNDLE}"
echo "[RuntimeExit] base_model_name_or_path=${RUNTIME_EXIT_BASE_MODEL_NAME_OR_PATH}"
echo "[RuntimeExit] results_dir=${RESULTS_DIR}"
echo "[RuntimeExit] datasets=${RUNTIME_EXIT_DATASETS}"
echo "[RuntimeExit] exit_layers=${RUNTIME_EXIT_LAYERS} threshold=${RUNTIME_EXIT_THRESHOLD}"
echo "[RuntimeExit] active_compaction_batch_size=${RUNTIME_EXIT_ACTIVE_COMPACTION_BATCH_SIZE}"
echo "[RuntimeExit] weight_quantization=${RUNTIME_EXIT_WEIGHT_QUANTIZATION} quantize_lm_head=${RUNTIME_EXIT_QUANTIZE_LM_HEAD}"
if [ -n "${RUNTIME_EXIT_DATASET_THRESHOLDS}" ]; then
  echo "[RuntimeExit] dataset_thresholds=${RUNTIME_EXIT_DATASET_THRESHOLDS}"
fi
if [ -n "${RUNTIME_EXIT_CONTROLLER_JSON}" ]; then
  echo "[RuntimeExit] controller_json=${RUNTIME_EXIT_CONTROLLER_JSON}"
  echo "[RuntimeExit] controller_decision_threshold=${RUNTIME_EXIT_CONTROLLER_DECISION_THRESHOLD}"
fi

"${PYTHON_BIN}" "${RUNTIME_EXIT_SCRIPT}" \
  --deploy_bundle "${DEPLOY_BUNDLE}" \
  --base_model_name_or_path "${RUNTIME_EXIT_BASE_MODEL_NAME_OR_PATH}" \
  --test_data_root "${TEST_DATA_ROOT}" \
  --datasets "${RUNTIME_EXIT_DATASETS}" \
  --output_dir "${RESULTS_DIR}" \
  --max_samples "${RUNTIME_EXIT_MAX_SAMPLES}" \
  --exit_layers "${RUNTIME_EXIT_LAYERS}" \
  --threshold "${RUNTIME_EXIT_THRESHOLD}" \
  --dataset_thresholds "${RUNTIME_EXIT_DATASET_THRESHOLDS}" \
  --controller_json "${RUNTIME_EXIT_CONTROLLER_JSON}" \
  --controller_decision_threshold "${RUNTIME_EXIT_CONTROLLER_DECISION_THRESHOLD}" \
  --length_norm "${RUNTIME_EXIT_LENGTH_NORM}" \
  --temperature "${RUNTIME_EXIT_TEMPERATURE}" \
  --active_compaction_batch_size "${RUNTIME_EXIT_ACTIVE_COMPACTION_BATCH_SIZE}" \
  --run_full_baseline "${RUNTIME_EXIT_RUN_FULL_BASELINE}" \
  --use_quant_bank_int4 "${RUNTIME_EXIT_USE_QUANT_BANK_INT4}" \
  --weight_quantization "${RUNTIME_EXIT_WEIGHT_QUANTIZATION}" \
  --quantize_lm_head "${RUNTIME_EXIT_QUANTIZE_LM_HEAD}" \
  --save_quantized_checkpoint "${RUNTIME_EXIT_SAVE_QUANTIZED_CHECKPOINT}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --trust_remote_code "${TRUST_REMOTE_CODE}" \
  --save_records "${RUNTIME_EXIT_SAVE_RECORDS}" \
  --warmup_samples "${RUNTIME_EXIT_WARMUP_SAMPLES}"

echo "[RuntimeExit] done: ${RESULTS_DIR}"
