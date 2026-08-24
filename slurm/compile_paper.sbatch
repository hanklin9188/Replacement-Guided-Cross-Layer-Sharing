#!/usr/bin/env bash
#SBATCH --job-name=rgcls_latex
#SBATCH --account=MST114566
#SBATCH --partition=dev
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gpus-per-node=H100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:?submit this job from the repository root}
cd "${REPO_ROOT}/paper"
PDFLATEX=${PDFLATEX:-pdflatex}
BIBTEX=${BIBTEX:-bibtex}
command -v "${PDFLATEX}" >/dev/null
command -v "${BIBTEX}" >/dev/null
"${PDFLATEX}" -interaction=nonstopmode -halt-on-error main.tex
"${BIBTEX}" main
"${PDFLATEX}" -interaction=nonstopmode -halt-on-error main.tex
"${PDFLATEX}" -interaction=nonstopmode -halt-on-error main.tex
test -s main.pdf
