#!/usr/bin/env bash
#SBATCH --job-name=ic27_base
#SBATCH --account=mst114566
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
: "${BACKBONE:?BACKBONE is required}"
VALID_TOKENS=${VALID_TOKENS:-8192}
rgcls cache-baseline --backbone "${BACKBONE}" --valid-tokens "${VALID_TOKENS}"
