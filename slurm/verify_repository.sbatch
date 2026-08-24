#!/usr/bin/env bash
#SBATCH --job-name=rgcls_verify
#SBATCH --account=MST114566
#SBATCH --partition=dev
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gpus-per-node=H100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:?submit this job from the repository root}
cd "${REPO_ROOT}"
test -f scripts/verify_repository.py
module load miniconda3/24.11.1
PYTHON_BIN=${PYTHON_BIN:-python}
"${PYTHON_BIN}" scripts/verify_repository.py
