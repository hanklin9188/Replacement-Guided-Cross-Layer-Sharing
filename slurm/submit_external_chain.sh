#!/usr/bin/env bash
set -euo pipefail

IMPORT_JOB=$(sbatch --parsable slurm/import_external_baselines.sbatch)
STATS_JOB=$(sbatch --parsable --dependency="afterok:${IMPORT_JOB}" slurm/external_statistics.sbatch)
VERIFY_JOB=$(sbatch --parsable --dependency="afterok:${STATS_JOB}" slurm/verify_repository.sbatch)
printf 'import=%s\nstatistics_and_quantization=%s\nverify=%s\n' "${IMPORT_JOB}" "${STATS_JOB}" "${VERIFY_JOB}"
