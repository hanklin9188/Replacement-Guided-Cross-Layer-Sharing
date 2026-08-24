#!/usr/bin/env bash
#SBATCH --job-name=ic27_geomatrix
#SBATCH --account=mst114566
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --gpus-per-node=1
#SBATCH --array=0-14%4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
: "${BACKBONE:?BACKBONE is required}"
: "${K:?K is required}"
POLICIES=(full random_0 random_1 random_2 random_3 random_4 adjacent weight_distance input_cosine \
  one_way_replacement directional_mean single_link symmetric_representative no_input_filter no_depth_constraints)
POLICY=${POLICIES[$SLURM_ARRAY_TASK_ID]}
rgcls geometry --backbone "${BACKBONE}" --valid-tokens "${VALID_TOKENS:-8192}" \
  --k "${K}" --policy "${POLICY}"
