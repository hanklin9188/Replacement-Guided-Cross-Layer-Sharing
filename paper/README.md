# Paper source

`main.tex` is the current ICASSP 2027 manuscript. It includes a compile-time
anonymous switch, a complete bibliography, and path-relative figures.

## Scheduled build on NCHC

From the repository root:

```bash
verify_job=$(sbatch --parsable slurm/verify_repository.sbatch)
sbatch --dependency="afterok:${verify_job}" slurm/compile_paper.sbatch
```

Do not compile on the login node.

## Local build outside the cluster

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

For an anonymous build:

```bash
pdflatex -jobname=main_anonymous '\def\ANONYMOUS{1}\input{main.tex}'
```

The cross-method quantization figure and its compact observed source table are
included. Regenerate it from the repository root with:

```bash
python scripts/generate_quantization_figure.py
```

The bundled `spconf.sty` and `IEEEbib.bst` come from the 2026 IEEE SPS ICIP
author kit as a provisional compile dependency. Compare them with the official
ICASSP 2027 author kit before submission. ICASSP 2027 currently specifies four
pages of technical content plus an optional references-only fifth page:
<https://2027.ieeeicassp.org/publishing-and-paper-presentation-options/>.
