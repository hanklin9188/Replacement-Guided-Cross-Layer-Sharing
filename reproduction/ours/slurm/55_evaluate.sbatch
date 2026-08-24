#!/usr/bin/env bash
#SBATCH --job-name=ic27_eval
#SBATCH --account=mst114566
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
: "${BACKBONE:?BACKBONE is required}"
: "${ROLE:?ROLE is required}"
: "${OUTPUT:?OUTPUT is required}"
ARGS=(evaluate --backbone "${BACKBONE}" --role "${ROLE}" --output "${OUTPUT}")
if [[ "${ROLE}" == step0 || "${ROLE}" == student ]]; then
  : "${K:?K is required for shared roles}"
  ARGS+=(--valid-tokens "${VALID_TOKENS:-8192}" --k "${K}" --policy "${POLICY:-full}")
fi
if [[ "${ROLE}" == student ]]; then
  : "${CHECKPOINT:?CHECKPOINT is required for student}"
  ARGS+=(--checkpoint "${CHECKPOINT}")
fi
rgcls "${ARGS[@]}"
