#!/usr/bin/env bash
#SBATCH --job-name=ic27_group
#SBATCH --account=mst114566
#SBATCH --partition=dev
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
: "${BACKBONE:?BACKBONE is required}"
: "${BUDGETS:?BUDGETS is required (comma-separated)}"
VALID_TOKENS=${VALID_TOKENS:-8192}
rgcls consolidate --backbone "${BACKBONE}" --valid-tokens "${VALID_TOKENS}"
IFS=',' read -r -a K_VALUES <<< "${BUDGETS}"
for K_VALUE in "${K_VALUES[@]}"; do
  rgcls group --backbone "${BACKBONE}" --valid-tokens "${VALID_TOKENS}" --k "${K_VALUE}"
  rgcls plot-replacement --backbone "${BACKBONE}" --valid-tokens "${VALID_TOKENS}" --k "${K_VALUE}"
done
