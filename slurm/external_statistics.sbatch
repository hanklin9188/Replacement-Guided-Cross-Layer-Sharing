#!/usr/bin/env bash
#SBATCH --job-name=rgcls_bootstrap
#SBATCH --account=MST114566
#SBATCH --partition=dev
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=H100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:?submit this job from the repository root}
OURS_PREDICTIONS=${OURS_PREDICTIONS:?set the full Ours predictions.csv path}
PYTHON_BIN=${PYTHON_BIN:-python3}
cd "${REPO_ROOT}"
"${PYTHON_BIN}" scripts/paired_bootstrap.py \
  --ours "${OURS_PREDICTIONS}" \
  --basis data/external/basis_sharing/incoming/predictions.csv \
  --svd data/external/svd_llm/incoming/predictions.csv \
  --output data/processed/paired_bootstrap_results.csv \
  --coverage-output data/processed/paired_bootstrap_task_coverage.csv \
  --samples 10000 \
  --seed 2027
"${PYTHON_BIN}" scripts/generate_quantization_figure.py
