# Reproducibility boundary

This document defines what a clean checkout can verify, what requires external
artifacts, and which evidence supports public claims.

## Tier A — public integrity, no GPU

```bash
python scripts/verify_repository.py
```

The verifier recomputes Ours run macros and seed statistics from compact
per-task rows, checks structural identities, validates paired-example coverage,
checks citations and links, and rejects private-path or credential markers.

## Tier B — scheduled reference smoke test

On NCHC, bootstrap and smoke-test only through Slurm:

```bash
bootstrap_job=$(sbatch --parsable reproduction/ours/slurm/00_bootstrap.sbatch)
sbatch --dependency="afterok:${bootstrap_job}" reproduction/ours/slurm/01_smoke.sbatch
```

The reference smoke fixture validates code paths and serialization contracts;
it is not evidence for the paper's quality claims.

## Tier C — paper-scale GPU reproduction

Obtain Meta Llama weights, the seven benchmark datasets, fixed recovery splits,
teachers, atlases, and step-0 policies under their respective terms. Fill
`configs/paper_models.tsv` from the provided example and submit the smoke/full
dependency chain in `docs/REPRODUCTION.md`.

## Controlled baseline path

The public tree includes the exact controlled Basis Sharing and SVD-LLM ports,
paper-matched recovery code, portable config templates, 4xH200 launchers,
quantization/evaluation scripts, and compact formal outputs. Re-running model
work still requires separately licensed weights, teachers, splits, and
checkpoints.

## Included

- modular and paper-era Ours code snapshots;
- explicit NCHC Slurm entry points and resource requests;
- model revisions, prototype budgets, and paper recovery settings;
- compact Ours data and structural observations;
- frozen all-method paper table, processed paired statistics, and complete
  cross-method quantization summaries;
- paper source, bibliography, figures, and executable integrity checks.

## Not redistributed

- Meta Llama weights, teachers, or recovered checkpoints;
- full benchmark corpora and private recovery splits;
- complete per-example prediction payloads;
- private cluster paths, credentials, caches, or scheduler logs;
- multi-GiB packed quantized checkpoints and raw scheduler work directories.

## Evidence rule

A public numeric claim should trace through:

```text
raw evaluator or structural artifact
  -> compact sanitized CSV
    -> scripts/verify_repository.py
      -> README / docs / manuscript
```

If a prose statement disagrees with the frozen paper table, the table and audit
note take precedence. Failed or incomplete jobs must not be promoted into a
headline result.
