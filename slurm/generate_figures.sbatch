#!/usr/bin/env bash
#SBATCH --job-name=rgcls_figures
#SBATCH --account=MST114566
#SBATCH --partition=dev
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=H100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:?submit this job from the repository root}
PYTHON_BIN=${PYTHON_BIN:-python3}
cd "${REPO_ROOT}"
export MPLBACKEND=Agg
"${PYTHON_BIN}" scripts/generate_paper_figures.py
