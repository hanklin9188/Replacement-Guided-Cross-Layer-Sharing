#!/usr/bin/env bash
#SBATCH --job-name=rgcls_paper_preview
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
mkdir -p assets/previews
convert -density 200 'paper/main.pdf[0]' -background white -alpha remove \
    assets/previews/paper_page-1.png
convert -density 200 'paper/main.pdf[3]' -background white -alpha remove \
    assets/previews/paper_page-4.png
test -s assets/previews/paper_page-1.png
test -s assets/previews/paper_page-4.png
echo "paper preview PASS"
