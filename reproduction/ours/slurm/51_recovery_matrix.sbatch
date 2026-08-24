#!/usr/bin/env bash
#SBATCH --job-name=ic27_matrix
#SBATCH --account=mst114566
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --gpus-per-node=1
#SBATCH --array=0-26%4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
: "${BACKBONE:?BACKBONE is required}"
: "${K:?K is required}"
POLICY=full
VARIANT=teacher_scaled_alignment
SEED=2027
SUPERVISED=final
case "${SLURM_ARRAY_TASK_ID}" in
  0|1|2|3|4) POLICY="random_${SLURM_ARRAY_TASK_ID}" ;;
  5) POLICY=adjacent ;;
  6) POLICY=weight_distance ;;
  7) POLICY=input_cosine; VARIANT=ce ;;
  8|9|10) POLICY=input_cosine; SEED=$((2019 + SLURM_ARRAY_TASK_ID)) ;;
  11) POLICY=one_way_replacement ;;
  12|13|14) VARIANT=ce; SEED=$((2015 + SLURM_ARRAY_TASK_ID)) ;;
  15|16|17) SEED=$((2012 + SLURM_ARRAY_TASK_ID)) ;;
  18) POLICY=directional_mean ;;
  19) POLICY=single_link ;;
  20) POLICY=symmetric_representative ;;
  21) POLICY=no_input_filter ;;
  22) POLICY=no_depth_constraints ;;
  23) VARIANT=hidden_mse ;;
  24) VARIANT=raw_update_mse ;;
  25) VARIANT=projected_unscaled ;;
  26) SUPERVISED=all_shared ;;
  *) exit 2 ;;
esac
rgcls recover --backbone "${BACKBONE}" --valid-tokens "${VALID_TOKENS:-8192}" --k "${K}" \
  --policy "${POLICY}" --variant "${VARIANT}" --seed "${SEED}" --rank 0 --lambda-align -1 \
  --supervised "${SUPERVISED}"
