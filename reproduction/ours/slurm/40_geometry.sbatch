#!/usr/bin/env bash
#SBATCH --job-name=ic27_geometry
#SBATCH --account=mst114566
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
: "${BACKBONE:?BACKBONE is required}"
: "${K:?K is required}"
VALID_TOKENS=${VALID_TOKENS:-8192}
POLICY=${POLICY:-full}
rgcls geometry --backbone "${BACKBONE}" --valid-tokens "${VALID_TOKENS}" --k "${K}" --policy "${POLICY}"
