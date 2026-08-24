#!/usr/bin/env bash
set -euo pipefail

budget="${1:?usage: run_adaptive_exit_budget.sh 15|20|25|30}"
case "${budget}" in
  15) prototypes=22 ;;
  20) prototypes=19 ;;
  25) prototypes=17 ;;
  30) prototypes=15 ;;
  *) echo "budget must be 15, 20, 25 or 30" >&2; exit 2 ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
workspace="${FAD_WORKSPACE_ROOT:?set FAD_WORKSPACE_ROOT}"
deploy_bundle="${FAD_DEPLOY_BUNDLE:?set FAD_DEPLOY_BUNDLE to deploy_bundle.pt}"
test_data="${FAD_TEST_DATA_ROOT:?set FAD_TEST_DATA_ROOT}"
policy="${FAD_EXIT_POLICY:-${repo_root}/data/raw/adaptive_exit/${budget}pct/oracle_distilled_exit_policy.json}"
results="${FAD_AE_RESULTS_DIR:-${workspace}/results/fad_ae_${budget}pct_$(date +%Y%m%d_%H%M%S)}"

PROJECT_ROOT="$(cd "${script_dir}/.." && pwd)" \
WORKSPACE_ROOT="${workspace}" \
DEPLOY_BUNDLE="${deploy_bundle}" \
TEST_DATA_ROOT="${test_data}" \
RESULTS_DIR="${results}" \
RUNTIME_EXIT_DATASETS="piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa" \
RUNTIME_EXIT_LAYERS="16,20,24" \
RUNTIME_EXIT_THRESHOLD=0.90 \
RUNTIME_EXIT_CONTROLLER_JSON="${policy}" \
RUNTIME_EXIT_CONTROLLER_DECISION_THRESHOLD=-1 \
RUNTIME_EXIT_ACTIVE_COMPACTION_BATCH_SIZE="${FAD_COMPACTION_BATCH_SIZE:-1}" \
RUNTIME_EXIT_RUN_FULL_BASELINE=True \
RUNTIME_EXIT_SAVE_RECORDS="${FAD_SAVE_RECORDS:-False}" \
SEED=44 \
bash "${script_dir}/run_runtime_conf_exit_final_llama.sh"

printf 'completed budget=%s prototypes=%s results=%s\n' "${budget}" "${prototypes}" "${results}"
