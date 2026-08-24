#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${PROJECT_DIR}"
VALID_TOKENS=8192

submit() {
  sbatch --parsable "$@"
}

if [[ -n "${START_AFTER:-}" ]]; then
  BOOTSTRAP_JOB=reused
  SMOKE_JOB=reused
  PREFLIGHT_JOB=reused
  DSMOKE_JOB=${START_AFTER}
else
  BOOTSTRAP_JOB=$(submit reproduction/ours/slurm/00_bootstrap.sbatch)
  SMOKE_JOB=$(submit --dependency="afterok:${BOOTSTRAP_JOB}" reproduction/ours/slurm/01_smoke.sbatch)
  PREFLIGHT_JOB=$(submit --dependency="afterok:${SMOKE_JOB}" reproduction/ours/slurm/02_preflight.sbatch)
  DSMOKE_JOB=$(submit --dependency="afterok:${PREFLIGHT_JOB}" reproduction/ours/slurm/03_data_smoke.sbatch)
fi
DATA_JOB=$(submit --dependency="afterok:${DSMOKE_JOB}" reproduction/ours/slurm/10_prepare_data.sbatch)
TEACHER_JOB=$(submit --dependency="afterok:${DATA_JOB}" reproduction/ours/slurm/20_teacher.sbatch)

BASE3_JOB=$(submit --dependency="afterok:${DATA_JOB}" --export=ALL,BACKBONE=llama32_3b,VALID_TOKENS=${VALID_TOKENS} reproduction/ours/slurm/30_baseline.sbatch)
ROWS3_JOB=$(submit --dependency="afterok:${BASE3_JOB}" --array=0-27%4 --export=ALL,BACKBONE=llama32_3b,VALID_TOKENS=${VALID_TOKENS} reproduction/ours/slurm/31_replacement_rows.sbatch)
GROUP3_JOB=$(submit --dependency="afterok:${ROWS3_JOB}" --export=ALL,BACKBONE=llama32_3b,VALID_TOKENS=${VALID_TOKENS},BUDGETS=22\,19\,17 reproduction/ours/slurm/32_consolidate_group.sbatch)

BASE8_JOB=$(submit --dependency="afterok:${DATA_JOB}" --export=ALL,BACKBONE=llama31_8b,VALID_TOKENS=${VALID_TOKENS} reproduction/ours/slurm/30_baseline.sbatch)
ROWS8_JOB=$(submit --dependency="afterok:${BASE8_JOB}" --array=0-31%4 --export=ALL,BACKBONE=llama31_8b,VALID_TOKENS=${VALID_TOKENS} reproduction/ours/slurm/31_replacement_rows.sbatch)
GROUP8_JOB=$(submit --dependency="afterok:${ROWS8_JOB}" --export=ALL,BACKBONE=llama31_8b,VALID_TOKENS=${VALID_TOKENS},BUDGETS=25\,23\,20 reproduction/ours/slurm/32_consolidate_group.sbatch)

STEP3_JOB=$(submit --dependency="afterok:${GROUP3_JOB}" --export=ALL,BACKBONE=llama32_3b,VALID_TOKENS=${VALID_TOKENS},K=19 reproduction/ours/slurm/35_step0_matrix.sbatch)
STEP8_JOB=$(submit --dependency="afterok:${GROUP8_JOB}" --export=ALL,BACKBONE=llama31_8b,VALID_TOKENS=${VALID_TOKENS},K=23 reproduction/ours/slurm/35_step0_matrix.sbatch)

GEOM3_JOB=$(submit --dependency="afterok:${TEACHER_JOB}:${GROUP3_JOB}" --export=ALL,BACKBONE=llama32_3b,VALID_TOKENS=${VALID_TOKENS},K=19 reproduction/ours/slurm/41_geometry_matrix.sbatch)
GEOM8_JOB=$(submit --dependency="afterok:${TEACHER_JOB}:${GROUP8_JOB}" --export=ALL,BACKBONE=llama31_8b,VALID_TOKENS=${VALID_TOKENS},K=23 reproduction/ours/slurm/41_geometry_matrix.sbatch)
PILOT3_JOB=$(submit --dependency="afterok:${GEOM3_JOB}_0" --export=ALL,BACKBONE=llama32_3b,VALID_TOKENS=${VALID_TOKENS},K=19 reproduction/ours/slurm/49_hparam_pilot.sbatch)
PILOT8_JOB=$(submit --dependency="afterok:${GEOM8_JOB}_0" --export=ALL,BACKBONE=llama31_8b,VALID_TOKENS=${VALID_TOKENS},K=23 reproduction/ours/slurm/49_hparam_pilot.sbatch)
SELECT3_JOB=$(submit --dependency="afterok:${PILOT3_JOB}" --export=ALL,BACKBONE=llama32_3b,VALID_TOKENS=${VALID_TOKENS},K=19,POLICY=full reproduction/ours/slurm/45_select_hparams.sbatch)
SELECT8_JOB=$(submit --dependency="afterok:${PILOT8_JOB}" --export=ALL,BACKBONE=llama31_8b,VALID_TOKENS=${VALID_TOKENS},K=23,POLICY=full reproduction/ours/slurm/45_select_hparams.sbatch)

REC3_JOB=$(submit --dependency="afterok:${GEOM3_JOB}:${SELECT3_JOB}" --export=ALL,BACKBONE=llama32_3b,VALID_TOKENS=${VALID_TOKENS},K=19 reproduction/ours/slurm/51_recovery_matrix.sbatch)
REC8_JOB=$(submit --dependency="afterok:${GEOM8_JOB}:${SELECT8_JOB}" --export=ALL,BACKBONE=llama31_8b,VALID_TOKENS=${VALID_TOKENS},K=23 reproduction/ours/slurm/51_recovery_matrix.sbatch)

GEOM3_MILD=$(submit --dependency="afterok:${TEACHER_JOB}:${GROUP3_JOB}" --export=ALL,BACKBONE=llama32_3b,VALID_TOKENS=${VALID_TOKENS},K=22,POLICY=full reproduction/ours/slurm/40_geometry.sbatch)
GEOM3_AGGR=$(submit --dependency="afterok:${TEACHER_JOB}:${GROUP3_JOB}" --export=ALL,BACKBONE=llama32_3b,VALID_TOKENS=${VALID_TOKENS},K=17,POLICY=full reproduction/ours/slurm/40_geometry.sbatch)
GEOM8_MILD=$(submit --dependency="afterok:${TEACHER_JOB}:${GROUP8_JOB}" --export=ALL,BACKBONE=llama31_8b,VALID_TOKENS=${VALID_TOKENS},K=25,POLICY=full reproduction/ours/slurm/40_geometry.sbatch)
GEOM8_AGGR=$(submit --dependency="afterok:${TEACHER_JOB}:${GROUP8_JOB}" --export=ALL,BACKBONE=llama31_8b,VALID_TOKENS=${VALID_TOKENS},K=20,POLICY=full reproduction/ours/slurm/40_geometry.sbatch)
REC3_MILD=$(submit --dependency="afterok:${GEOM3_MILD}:${SELECT3_JOB}" --export=ALL,BACKBONE=llama32_3b,VALID_TOKENS=${VALID_TOKENS},K=22,POLICY=full,VARIANT=teacher_scaled_alignment,SEED=2027,RANK=0,LAMBDA_ALIGN=-1,SUPERVISED=final reproduction/ours/slurm/50_recovery.sbatch)
REC3_AGGR=$(submit --dependency="afterok:${GEOM3_AGGR}:${SELECT3_JOB}" --export=ALL,BACKBONE=llama32_3b,VALID_TOKENS=${VALID_TOKENS},K=17,POLICY=full,VARIANT=teacher_scaled_alignment,SEED=2027,RANK=0,LAMBDA_ALIGN=-1,SUPERVISED=final reproduction/ours/slurm/50_recovery.sbatch)
REC8_MILD=$(submit --dependency="afterok:${GEOM8_MILD}:${SELECT8_JOB}" --export=ALL,BACKBONE=llama31_8b,VALID_TOKENS=${VALID_TOKENS},K=25,POLICY=full,VARIANT=teacher_scaled_alignment,SEED=2027,RANK=0,LAMBDA_ALIGN=-1,SUPERVISED=final reproduction/ours/slurm/50_recovery.sbatch)
REC8_AGGR=$(submit --dependency="afterok:${GEOM8_AGGR}:${SELECT8_JOB}" --export=ALL,BACKBONE=llama31_8b,VALID_TOKENS=${VALID_TOKENS},K=20,POLICY=full,VARIANT=teacher_scaled_alignment,SEED=2027,RANK=0,LAMBDA_ALIGN=-1,SUPERVISED=final reproduction/ours/slurm/50_recovery.sbatch)

REPORT_DEPS="${REC3_JOB}:${REC8_JOB}:${REC3_MILD}:${REC3_AGGR}:${REC8_MILD}:${REC8_AGGR}:${STEP3_JOB}:${STEP8_JOB}"
REPORT_JOB=$(submit --dependency="afterok:${REPORT_DEPS}" reproduction/ours/slurm/60_report.sbatch)

printf 'bootstrap=%s\nsmoke=%s\npreflight=%s\ndata_smoke=%s\ndata=%s\nteacher=%s\n' "${BOOTSTRAP_JOB}" "${SMOKE_JOB}" "${PREFLIGHT_JOB}" "${DSMOKE_JOB}" "${DATA_JOB}" "${TEACHER_JOB}"
printf '3b: baseline=%s rows=%s groups=%s step0=%s geometry=%s pilot=%s select=%s recovery=%s\n' "${BASE3_JOB}" "${ROWS3_JOB}" "${GROUP3_JOB}" "${STEP3_JOB}" "${GEOM3_JOB}" "${PILOT3_JOB}" "${SELECT3_JOB}" "${REC3_JOB}"
printf '8b: baseline=%s rows=%s groups=%s step0=%s geometry=%s pilot=%s select=%s recovery=%s\n' "${BASE8_JOB}" "${ROWS8_JOB}" "${GROUP8_JOB}" "${STEP8_JOB}" "${GEOM8_JOB}" "${PILOT8_JOB}" "${SELECT8_JOB}" "${REC8_JOB}"
printf 'report=%s\n' "${REPORT_JOB}"
