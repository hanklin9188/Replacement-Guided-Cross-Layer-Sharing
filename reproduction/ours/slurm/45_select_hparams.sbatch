#!/usr/bin/env bash
#SBATCH --job-name=ic27_select
#SBATCH --account=mst114566
#SBATCH --partition=dev
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
: "${BACKBONE:?BACKBONE is required}"
: "${K:?K is required}"
rgcls select-hparams --backbone "${BACKBONE}" --valid-tokens "${VALID_TOKENS:-8192}" \
  --k "${K}" --policy "${POLICY:-full}"
