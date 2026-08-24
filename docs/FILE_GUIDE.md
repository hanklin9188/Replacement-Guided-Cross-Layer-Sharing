# File guide

| Path | Purpose |
|---|---|
| `paper/main.tex` | Current ICASSP 2027 manuscript with anonymous-build switch. |
| `paper/references.bib` | Complete bibliography for every cited key. |
| `paper/Figure/` | Manuscript-facing framework and structural figures. |
| `assets/figures/` | README previews and multi-format analysis figures. |
| `src/icassp27/` | Modular reference implementation of replacement, grouping, recovery, evaluation, and reporting. |
| `configs/experiment.yaml` | Portable model revisions, budgets, method settings, and smoke defaults. |
| `configs/paper_models.example.tsv` | Six-row exact paper recovery path contract. |
| `reproduction/ours/core/` | Paper-era functional observation, grouping, recovery, evaluation, and quantization snapshot. |
| `reproduction/ours/scripts/` | Policy preparation and exact CE/CE+KD recovery launchers. |
| `reproduction/ours/slurm/` | Resource-explicit reference and paper-scale jobs. |
| `reproduction/baselines/` | Complete reserved integration tree for Basis Sharing and SVD-LLM. |
| `data/ours/per_task.csv` | 294 compact task rows underlying 42 current Ours run macros. |
| `data/ours/main_summary.csv` | Current Ours Pure/CE/CE+KD mean and sample SD. |
| `data/ours/directed_costs.csv` | 1,748 finite off-diagonal directed replacement observations. |
| `data/ours/pair_analysis.csv` | 874 mutual costs and directional asymmetries. |
| `data/ours/group_analysis.csv` | Eleven selected non-singleton groups and envelope quantities. |
| `data/ours/joint_analysis.csv` | Six simultaneous-deployment distortion rows. |
| `data/ours/structural_ablation.csv` | Five 3B--20% structure variants and equivalence labels. |
| `data/ours/quantization.csv` | Ours 8B--15/25% BF16/W8A16/W4A16 results. |
| `data/processed/paper_main_table.csv` | Frozen manuscript aggregate for all methods. |
| `data/processed/paired_bootstrap_results.csv` | Synchronized paired comparisons with confidence intervals. |
| `data/processed/paired_bootstrap_task_coverage.csv` | Exact question-alignment audit. |
| `data/processed/serialized_byte_reduction_combined.csv` | Path-free standalone byte accounting for all methods. |
| `data/external/schema/` | Baseline payload templates and manifest schema. |
| `scripts/verify_repository.py` | Standard-library numeric, citation, link, and hygiene authority. |
| `scripts/import_external_baselines.py` | Scheduled external payload validator and hasher. |
| `scripts/generate_paper_figures.py` | Rebuilds data-driven non-quantization figures. |
| `scripts/generate_quantization_figure.py` | Refuses output until all three methods are complete. |
| `docs/github-actions/verify.yml` | Ready-to-enable workflow for push and pull-request audits. |
