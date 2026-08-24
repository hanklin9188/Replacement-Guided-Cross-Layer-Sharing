#!/usr/bin/env bash
#SBATCH --job-name=mr4h200
#SBATCH --partition=8gpus
#SBATCH --account=mst114566
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:H200:4
#SBATCH --exclude=25a-hgpn144
#SBATCH --mem=500G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

: "${RGCLS_REPO_ROOT:=${SLURM_SUBMIT_DIR:?submit from the repository root}}"
: "${RGCLS_PYTHON_ENV:=$RGCLS_REPO_ROOT/.venv}"
: "${METHOD_CONFIG:=$RGCLS_REPO_ROOT/configs/method_recovery_4h200.example.yaml}"
: "${METHOD_SPEC:?METHOD_SPEC must point to an immutable JSON spec}"

module load miniconda3/26.1.1
export PYTHONPATH="$RGCLS_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$RGCLS_REPO_ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true
export OMP_NUM_THREADS=8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p "$RGCLS_REPO_ROOT/logs"

set +e
srun --nodes=1 --ntasks=1 --gpus-per-task=4 --cpus-per-task=48 \
  "$RGCLS_PYTHON_ENV/bin/torchrun" --standalone --nnodes=1 --nproc_per_node=4 \
  -m icassp27.controlled_baselines.method_cli --config "$METHOD_CONFIG" --spec "$METHOD_SPEC"
run_status=$?
set -e

coordinator_status=0
if [[ -n "${METHOD_MANIFEST:-}" ]]; then
  "$RGCLS_PYTHON_ENV/bin/python" "$RGCLS_REPO_ROOT/reproduction/baselines/scripts/method_recovery_coordinator.py" \
    --manifest "$METHOD_MANIFEST" --completed-spec "$METHOD_SPEC" --exit-code "$run_status" \
    || coordinator_status=$?
fi

if (( run_status != 0 )); then
  exit "$run_status"
fi
exit "$coordinator_status"
