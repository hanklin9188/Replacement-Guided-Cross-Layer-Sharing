#!/usr/bin/env bash
#SBATCH --job-name=rgcls_struct_figure
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
module load miniconda3/24.11.1
export MPLBACKEND=Agg
python -c "from scripts.generate_paper_figures import plot_structural_validation_combined; plot_structural_validation_combined()"

for suffix in pdf png svg; do
    test -s "assets/figures/diag_structural_validation.${suffix}"
done
cp -p assets/figures/diag_structural_validation.pdf \
    paper/Figure/diag_structural_validation.pdf

echo "structural-validation render PASS"
