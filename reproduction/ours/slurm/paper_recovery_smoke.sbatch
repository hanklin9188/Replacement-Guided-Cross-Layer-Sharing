#!/usr/bin/env bash
#SBATCH --job-name=rgcls_rec_smoke
#SBATCH --account=MST114566
#SBATCH --partition=dev
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=H200:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=320G
#SBATCH --array=0-1%1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:?submit from the repository root}
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID * 3))
bash "${REPO_ROOT}/reproduction/ours/scripts/run_paper_recovery.sh" "${MODEL_INDEX}" kd 42 smoke
