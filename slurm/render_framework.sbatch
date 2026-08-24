#!/usr/bin/env bash
#SBATCH --job-name=rgcls_framework
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
test -s paper/Figure/replace.pdf
if command -v pdftoppm >/dev/null 2>&1; then
  pdftoppm -png -r 180 -singlefile paper/Figure/replace.pdf assets/figures/framework
elif command -v magick >/dev/null 2>&1; then
  magick -density 180 'paper/Figure/replace.pdf[0]' assets/figures/framework.png
elif command -v gs >/dev/null 2>&1; then
  gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=pngalpha -r180 \
    -dFirstPage=1 -dLastPage=1 \
    -sOutputFile=assets/figures/framework.png paper/Figure/replace.pdf
else
  module load miniconda3/24.11.1
  PYTHON_BIN=${PYTHON_BIN:-python}
  if ! "${PYTHON_BIN}" -c 'import fitz' >/dev/null 2>&1; then
    echo 'No PDF rasterizer found. Install requirements-analysis.txt or set PYTHON_BIN.' >&2
    exit 1
  fi
  "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import fitz

root = Path.cwd()
document = fitz.open(root / "paper/Figure/replace.pdf")
page = document.load_page(0)
page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False).save(
    root / "assets/figures/framework.png"
)
PY
fi
test -s assets/figures/framework.png
