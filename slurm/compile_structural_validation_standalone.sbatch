#!/usr/bin/env bash
#SBATCH --job-name=rgcls_struct_pdf
#SBATCH --account=MST114566
#SBATCH --partition=dev
#SBATCH --time=00:05:00
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

cd paper
PDFLATEX=${PDFLATEX:-pdflatex}
"${PDFLATEX}" -interaction=nonstopmode -halt-on-error \
    -jobname=structural_validation \
    -output-directory=Figure \
    Figure/diag_structural_validation_standalone.tex
test -s Figure/structural_validation.pdf
grep -q 'Output written on Figure/structural_validation.pdf (1 page' \
    Figure/structural_validation.log

mkdir -p ../assets/previews
convert -density 200 'Figure/structural_validation.pdf[0]' \
    -background white -alpha remove \
    ../assets/previews/structural_validation.png
test -s ../assets/previews/structural_validation.png

echo "standalone structural-validation TikZ PDF PASS"
