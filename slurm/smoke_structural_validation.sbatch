#!/usr/bin/env bash
#SBATCH --job-name=rgcls_struct_smoke
#SBATCH --account=MST114566
#SBATCH --partition=dev
#SBATCH --time=00:02:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gpus-per-node=H100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:?submit this job from the repository root}
cd "${REPO_ROOT}"
module load miniconda3/24.11.1
export MPLBACKEND=Agg
python -c "from scripts.generate_paper_figures import plot_structural_validation_combined; print('structural-validation smoke PASS')"
