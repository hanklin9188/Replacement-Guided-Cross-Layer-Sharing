#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
suite_id="${1:-portfolio_$(date +%Y%m%d_%H%M%S)}"

for spec in "15 22" "20 19" "25 17" "30 15"; do
  read -r budget prototypes <<< "${spec}"
  run_name="3b_b${budget}_k${prototypes}_s44_fad"
  "${script_dir}/run_latest_fad_train.sh" \
    llama32_3b "${budget}" "${prototypes}" 44 fad velocity_pca \
    "${run_name}" "${suite_id}"
done
