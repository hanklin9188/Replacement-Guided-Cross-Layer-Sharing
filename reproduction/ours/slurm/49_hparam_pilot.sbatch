#!/usr/bin/env bash
#SBATCH --job-name=ic27_pilot
#SBATCH --account=mst114566
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --gpus-per-node=1
#SBATCH --array=0-14%4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
: "${BACKBONE:?BACKBONE is required}"
: "${K:?K is required}"
RANKS=(64 128 256)
LAMBDAS=(0.03 0.1 0.3 1.0 3.0)
RANK_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
LAMBDA_INDEX=$((SLURM_ARRAY_TASK_ID % 5))
rgcls recover --backbone "${BACKBONE}" --valid-tokens "${VALID_TOKENS:-8192}" --k "${K}" \
  --policy full --variant teacher_scaled_alignment --seed 2027 --rank "${RANKS[$RANK_INDEX]}" \
  --lambda-align "${LAMBDAS[$LAMBDA_INDEX]}" --supervised final --selection-only
