#!/usr/bin/env bash
#SBATCH --job-name=cb1h200
#SBATCH --partition=8gpus
#SBATCH --account=mst114566
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

: "${RGCLS_REPO_ROOT:=${SLURM_SUBMIT_DIR:?submit from the repository root}}"
: "${RGCLS_PYTHON_ENV:=$RGCLS_REPO_ROOT/.venv}"
: "${CONTROLLED_CONFIG:=$RGCLS_REPO_ROOT/configs/controlled_baselines.example.yaml}"
: "${CONTROLLED_SPEC:?CONTROLLED_SPEC must point to one immutable JSON spec}"

module load miniconda3/26.1.1
export PYTHONPATH="$RGCLS_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$RGCLS_REPO_ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true
export OMP_NUM_THREADS=8
mkdir -p "$RGCLS_REPO_ROOT/logs"

set +e
srun --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=12 \
  "$RGCLS_PYTHON_ENV/bin/torchrun" --standalone --nnodes=1 --nproc_per_node=1 \
  -m icassp27.controlled_baselines.cli --config "$CONTROLLED_CONFIG" run-spec --spec "$CONTROLLED_SPEC"
run_status=$?
set -e

chain_status=0
if [[ -n "${CONTROLLED_CHAIN_MANIFEST:-}" ]]; then
  "$RGCLS_PYTHON_ENV/bin/python" "$RGCLS_REPO_ROOT/reproduction/baselines/scripts/submit_controlled_chain_next.py" \
    --manifest "$CONTROLLED_CHAIN_MANIFEST" \
    --current-index "${CONTROLLED_CHAIN_INDEX:?CONTROLLED_CHAIN_INDEX is required for chained jobs}" \
    || chain_status=$?
fi

if (( run_status != 0 )); then
  exit "$run_status"
fi
exit "$chain_status"
