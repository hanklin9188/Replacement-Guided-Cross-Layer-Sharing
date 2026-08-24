# Reproduction guide

## NCHC safety boundary

Do not run model loading, training, evaluation, plotting, compilation, or even
small smoke tests on a login node. Submit every such action with `sbatch`.
Login-node activity should be limited to editing, `sbatch`, `squeue`, `sacct`,
`scontrol`, `scancel`, and small read-only inspections.

All supplied jobs use account `MST114566`, explicit partitions, time, CPUs,
memory, GPUs where needed, and `logs/` output paths. Adjust the account only if
running outside the original allocation.

## Verification and paper build

From the repository root:

```bash
verify_job=$(sbatch --parsable slurm/verify_repository.sbatch)
sbatch --dependency="afterok:${verify_job}" slurm/compile_paper.sbatch
```

The compile job expects `pdflatex` and `bibtex` on `PATH`; alternatively export
`PDFLATEX` and `BIBTEX` before submission.

## Modular reference pipeline

Bootstrap the isolated environment, then gate the smoke test on success:

```bash
bootstrap_job=$(sbatch --parsable reproduction/ours/slurm/00_bootstrap.sbatch)
sbatch --dependency="afterok:${bootstrap_job}" reproduction/ours/slurm/01_smoke.sbatch
```

Model revisions and final prototype budgets are recorded in
`configs/experiment.yaml`. Full structural jobs are under
`reproduction/ours/slurm/`.

## Exact paper recovery matrix

Copy `configs/paper_models.example.tsv` to the ignored
`configs/paper_models.tsv` and replace every angle-bracket placeholder with a
real external path. Required columns include the immutable step-0 checkpoint,
selected policy, atlas, teacher, and fixed train/validation split for each of
the six operating points.

Export these paths before submission:

```bash
export RGCLS_WORKSPACE_ROOT=/path/to/writable/workspace
export RGCLS_PYTHON_ENV_BIN=/path/to/conda/env/bin
export RGCLS_TEST_DATA_ROOT=/path/to/seven_task_datasets
export RGCLS_CACHE_ROOT=/path/to/huggingface/cache
```

First run the two-item 3B/8B CE+KD smoke array. Submit the full 36-item matrix
only with an `afterok` dependency:

```bash
smoke_job=$(sbatch --parsable reproduction/ours/slurm/paper_recovery_smoke.sbatch)
sbatch --dependency="afterok:${smoke_job}" reproduction/ours/slurm/paper_recovery_full.sbatch
```

The full matrix is 6 operating points × CE/CE+KD × seeds 42/43/44, with at most
two concurrent array tasks. Each task requests 2 H200 GPUs, 24 CPUs, 320 GiB,
and 12 hours. The wrapper fixes:

- 3B: 7,500 updates, effective batch 32, shared-bank LR 3e-5, adapter LR 8e-5;
- 8B: 10,000 updates, effective batch 8, shared-bank LR 2.5e-5, adapter LR 6e-5;
- maximum length 384, warmup-cosine schedule, gradient clip 1.0;
- held-out decision-CE checkpoint selection every 500 updates;
- `length_norm=none` seven-task log-probability evaluation.

## Figure regeneration

With a Python environment containing NumPy and Matplotlib:

```bash
sbatch --export=ALL,PYTHON_BIN=/path/to/env/bin/python slurm/generate_figures.sbatch
```

The cross-method quantization figure is fully self-contained:

```bash
python scripts/generate_quantization_figure.py
```

Baseline compression/recovery/quantization commands and required external
artifact paths are documented in `docs/BASELINES.md`.

## Expected outputs

Preserve, outside Git when large:

- immutable policies and step-0 checkpoints;
- per-task evaluator JSON and per-example predictions;
- recovery checkpoint-selection reports;
- standalone serialization and quantization manifests;
- environment and upstream revisions;
- Slurm stdout/stderr and job IDs.

Only compact, sanitized derivatives belong in the public repository.
