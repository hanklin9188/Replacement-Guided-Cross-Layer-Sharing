#!/usr/bin/env bash
#SBATCH --job-name=rgcls_recovery
#SBATCH --account=MST114566
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=H200:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=320G
#SBATCH --array=0-35%2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:?submit from the repository root}
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / 6))
WITHIN=$((SLURM_ARRAY_TASK_ID % 6))
if (( WITHIN < 3 )); then
  OBJECTIVE=ce
  SEED=$((42 + WITHIN))
else
  OBJECTIVE=kd
  SEED=$((39 + WITHIN))
fi
bash "${REPO_ROOT}/reproduction/ours/scripts/run_paper_recovery.sh" "${MODEL_INDEX}" "${OBJECTIVE}" "${SEED}" full
