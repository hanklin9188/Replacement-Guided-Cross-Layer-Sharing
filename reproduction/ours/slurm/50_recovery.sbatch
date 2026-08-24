#!/usr/bin/env bash
#SBATCH --job-name=ic27_recovery
#SBATCH --account=mst114566
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
: "${BACKBONE:?BACKBONE is required}"
: "${K:?K is required}"
: "${VARIANT:?VARIANT is required}"
: "${SEED:?SEED is required}"
VALID_TOKENS=${VALID_TOKENS:-8192}
POLICY=${POLICY:-full}
RANK=${RANK:-128}
LAMBDA_ALIGN=${LAMBDA_ALIGN:-0.3}
SUPERVISED=${SUPERVISED:-final}
SELECTION_ONLY=${SELECTION_ONLY:-0}
EXTRA_ARGS=()
if [[ "${SELECTION_ONLY}" == 1 ]]; then
  EXTRA_ARGS+=(--selection-only)
fi
rgcls recover --backbone "${BACKBONE}" --valid-tokens "${VALID_TOKENS}" --k "${K}" \
  --policy "${POLICY}" --variant "${VARIANT}" --seed "${SEED}" --rank "${RANK}" \
  --lambda-align "${LAMBDA_ALIGN}" --supervised "${SUPERVISED}" "${EXTRA_ARGS[@]}"
