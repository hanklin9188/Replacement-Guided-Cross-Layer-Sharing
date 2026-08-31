#!/usr/bin/env bash
#SBATCH --job-name=rgcls_struct_tex
#SBATCH --account=MST114566
#SBATCH --partition=dev
#SBATCH --time=00:03:00
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
python scripts/export_structural_validation_pgfplots.py

for file in pair_3b pair_8b group_3b group_8b joint_3b joint_8b; do
    test -s "paper/Figure/structural_validation_data/${file}.csv"
done

echo "structural-validation PGFPlots data PASS"
