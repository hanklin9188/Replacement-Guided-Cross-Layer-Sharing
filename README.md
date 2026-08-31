<div align="center">

# Cross-Layer Replacement Sharing for Model Compression
### Budgeted LLM compression through directed functional replacement, fixed sharing structure, and recoverability-aware evaluation.

[![Verify repository](https://github.com/hanklin9188/Replacement-Guided-Cross-Layer-Sharing/actions/workflows/verify.yml/badge.svg)](https://github.com/hanklin9188/Replacement-Guided-Cross-Layer-Sharing/actions/workflows/verify.yml)
[![Paper](https://img.shields.io/badge/paper-ICASSP%202027-8A2BE2)](paper/Figure/overview-3.pdf)
[![Artifact](https://img.shields.io/badge/artifact-auditable-2f6f62)](REPRODUCIBILITY.md)
[![Baselines](https://img.shields.io/badge/Basis%20Sharing%20%2F%20SVD--LLM-reproducible-2f6f62)](docs/BASELINES.md)

[繁體中文](README_zh-TW.md) · [Paper PDF](paper/Figure/overview-3.pdf) · [Paper source](paper/README.md) · [Method](docs/METHOD.md) · [Results](docs/RESULTS.md) · [Reproduce](docs/REPRODUCTION.md) · [Baseline import](docs/BASELINES.md)

<img src="assets/figures/framework.png" alt="Replacement-guided cross-layer sharing framework" width="100%">

</div>

---

Cross-layer sharing saves storage by letting multiple decoder depths reuse a
smaller bank of feed-forward networks (FFNs). This project chooses that sharing
structure from **directed full-model replacement cost**: a donor FFN is tested
inside the frozen model at a target depth, complete-link grouping controls the
worst pair inside each group, and the final representative is selected in the
actual donor-to-target direction.

This repository is organized as a research artifact rather than a paper-code
dump. Ours code, compact observations, paper sources, figures, validation
scripts, and NCHC Slurm launchers are included. Basis Sharing and SVD-LLM have
strict, ready-to-fill integration slots because their full runs live on a
different server.

## Structural validation figure

<img src="assets/figures/diag_structural_validation.png" alt="Directed replacement asymmetry, group-bound validation, and joint-deployment distortion" width="100%">

The manuscript uses a native PGFPlots/TikZ version of this figure. Compact plot
tables are exported from `data/ours/pair_analysis.csv`,
`data/ours/group_analysis.csv`, and `data/ours/joint_analysis.csv`; the
standalone build then emits `paper/Figure/structural_validation.pdf`.

On NCHC, regenerate and validate it through the dependency-gated Slurm chain:

```bash
smoke_job=$(sbatch --parsable slurm/smoke_structural_validation.sbatch)
figure_job=$(sbatch --parsable --dependency="afterok:${smoke_job}" \
  slurm/compile_structural_validation_standalone.sbatch)
paper_job=$(sbatch --parsable --dependency="afterok:${figure_job}" \
  slurm/compile_paper.sbatch)
sbatch --dependency="afterok:${paper_job}" slurm/verify_repository.sbatch
```

## What this project demonstrates

- **Functional structural selection:** sharing is determined by end-to-end
  replacement behavior rather than weight distance alone.
- **Budgeted grouping:** a fixed number of stored FFNs is enforced through a
  complete-link minimax construction.
- **Directional representatives:** the selected donor minimizes its worst
  outgoing replacement cost inside the group.
- **Separated recovery:** Pure, CE, and CE+KD expose structure quality,
  label-only recoverability, and teacher-assisted stability separately.
- **Deployment accounting:** unique parameters, standalone serialized bytes,
  and BF16/INT8/INT4 behavior are reported without claiming inherent FLOP
  savings from sharing.

## Evidence snapshot

The frozen paper table uses seven-task macro accuracy over PIQA, Social-IQA,
WinoGrande, ARC-Challenge, ARC-Easy, HellaSwag, and OpenBookQA. CE and CE+KD
report mean ± sample SD over seeds 42/43/44.

| Backbone | Target | Ours CE+KD | Strongest baseline | Advantage |
|---|---:|---:|---:|---:|
| Llama-3.2-3B | 15% | **84.83 ± 0.47** | 71.28 | +13.55 pp |
| Llama-3.2-3B | 20% | **85.06 ± 0.08** | 68.15 | +16.91 pp |
| Llama-3.2-3B | 25% | **84.09 ± 0.18** | 65.06 | +19.03 pp |
| Llama-3.1-8B | 15% | **86.14 ± 0.33** | 76.43 | +9.71 pp |
| Llama-3.1-8B | 20% | **85.94 ± 0.34** | 73.89 | +12.05 pp |
| Llama-3.1-8B | 25% | **85.92 ± 0.36** | 69.96 | +15.96 pp |

At the selected 8B checkpoints, Ours INT8 preserves the BF16 result while
cutting standalone storage substantially:

| Structure | Precision | Macro accuracy | Serialized size |
|---|---|---:|---:|
| 15% | BF16 | 86.48% | 12.67 GiB |
| 15% | INT8 | **86.52%** | **7.32 GiB** |
| 25% | BF16 | 86.28% | 11.04 GiB |
| 25% | INT8 | **86.34%** | **6.50 GiB** |

The complete 15%/25% cross-method storage--accuracy figure, its 21 source
points, per-task baseline aggregates, paired bootstrap intervals, exact byte
manifests, and regeneration script are included.

<img src="assets/figures/quantization.png" alt="Cross-method quantized storage-accuracy trade-off" width="78%">

## Artifact status

| Area | Status |
|---|---|
| Ours method code and portable reference package | **Included** |
| Ours Pure / CE / CE+KD compact results | **Included and verifier-checked** |
| Directed, group, joint, and ablation analyses | **Included and verifier-checked** |
| Processed paired baseline comparisons and byte accounting | **Included** |
| Controlled Basis Sharing / SVD-LLM compression and recovery code | **Included** |
| Baseline configs, Slurm jobs, and quantization pipeline | **Included** |
| Cross-method quantization source tables and figure | **Included and verifier-checked** |
| Model weights and full datasets | **Intentionally excluded** |

## Verify without a GPU

The public integrity audit uses only the Python standard library:

```bash
python scripts/verify_repository.py
```

It recomputes all Ours macro means and seed standard deviations from the
released per-task rows, checks structural identities and paired-example
coverage, validates citations and local links, and rejects private cluster
paths or credentials. GitHub Actions runs the same audit on every push and
pull request.

On NCHC, submit the same check through Slurm:

```bash
sbatch slurm/verify_repository.sbatch
```

## Reproduce the research workflow

GPU work must be scheduled; do not run model code on a cluster login node.
Start with the dependency-gated reference pipeline:

```bash
bootstrap_job=$(sbatch --parsable reproduction/ours/slurm/00_bootstrap.sbatch)
sbatch --dependency="afterok:${bootstrap_job}" reproduction/ours/slurm/01_smoke.sbatch
```

The exact paper recovery matrix uses six operating points × two objectives ×
three seeds and requests two H200 GPUs per task:

```bash
cp configs/paper_models.example.tsv configs/paper_models.tsv
# Fill external artifact paths and export the variables documented in
# docs/REPRODUCTION.md, then submit the smoke test before the full array.
sbatch reproduction/ours/slurm/paper_recovery_smoke.sbatch
```

See [docs/REPRODUCTION.md](docs/REPRODUCTION.md) for the complete staged
workflow and [docs/BASELINES.md](docs/BASELINES.md) for the controlled baseline
compression, recovery, and quantization paths.

## Repository map

```text
paper/                       manuscript, bibliography, template, paper figures
src/icassp27/                modular reference implementation
reproduction/ours/           paper-era core snapshot, scripts, and Slurm jobs
reproduction/baselines/      controlled baseline runners, analysis, and Slurm jobs
data/ours/                   compact Ours observations and measurements
data/processed/              paper table, paired statistics, and byte accounting
data/external/               optional raw-payload import schemas
assets/figures/              README previews and publication figures
scripts/                     verification, plotting, and import utilities
docs/                        method, results, audit, reproduction, and file guide
```

## Five-minute reviewer path

1. Inspect the framework above and [method note](docs/METHOD.md).
2. Read the [frozen result table and provenance split](docs/RESULTS.md).
3. Run `python scripts/verify_repository.py`.
4. Inspect `data/ours/group_analysis.csv` and
   `data/ours/joint_analysis.csv`.
5. Read [LIMITATIONS.md](LIMITATIONS.md) before interpreting external baseline
   or deployment claims.

## Reproducibility boundary

Included: code and scheduler entry points, immutable configuration examples,
compact observed tables, figure inputs, processed paired statistics, paper
sources, and executable integrity checks.

Not redistributed: Meta Llama weights, recovered checkpoints, full benchmark
corpora, private cluster paths, credentials, raw scheduler logs, and multi-GiB
packed checkpoints. Upstream baseline revisions are pinned; their source may be
cloned under `third_party/` when provenance preflight is required.

## Citation and terms

Citation metadata is in [CITATION.cff](CITATION.cff). Original unpublished
material remains subject to [NOTICE.md](NOTICE.md); third-party models, datasets,
templates, and baseline projects retain their own terms as documented in
[THIRD_PARTY.md](THIRD_PARTY.md).
