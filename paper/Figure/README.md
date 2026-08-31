# Manuscript figures

- `replace.pdf`: framework supplied with the manuscript.
- `diag_structural_validation.tex`: native PGFPlots figure for directed
  asymmetry, group envelope, and joint-deployment diagnostics. Its compact
  CSV tables are generated from `data/ours/*_analysis.csv` by
  `scripts/export_structural_validation_pgfplots.py`.
- `diag_structural_validation_standalone.tex`: independently compilable wrapper
  for the native PGFPlots figure; the scheduled build emits
  `structural_validation.pdf`.
- `structural_validation_data/`: compact generated plot tables committed so the
  paper remains buildable without rerunning model analysis. Panel (c) stores
  $C^\star=C_{\mathrm{joint}}/L$, using 28 layers for 3B and 32 for 8B.
- `../../scripts/export_structural_validation_pgfplots.py`: standard-library
  exporter from the released analysis CSV files.
- `../../slurm/compile_structural_validation_standalone.sbatch`: scheduled
  exporter, standalone PDF build, and preview check for NCHC.
- `fig_quantization_15_25_storage_accuracy.png`: intentionally absent until
  complete Basis Sharing and SVD-LLM quantization payloads are validated.

Run the scheduled figure job documented in `docs/REPRODUCTION.md` after the
external import completes.
