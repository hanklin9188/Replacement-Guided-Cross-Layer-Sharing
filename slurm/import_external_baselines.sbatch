#!/usr/bin/env bash
#SBATCH --job-name=rgcls_import
#SBATCH --account=MST114566
#SBATCH --partition=dev
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=H100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:?submit this job from the repository root}
cd "${REPO_ROOT}"
module load miniconda3/24.11.1
PYTHON_BIN=${PYTHON_BIN:-python}
"${PYTHON_BIN}" scripts/import_external_baselines.py
