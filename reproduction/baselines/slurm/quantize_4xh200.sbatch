#!/usr/bin/env bash
#SBATCH --job-name=qb_smoke
#SBATCH --partition=8gpus
#SBATCH --account=mst114566
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=500G
#SBATCH --time=06:00:00
#SBATCH --exclude=25a-hgpn144
#SBATCH --output=logs/quant_baseline_%A_%a.out
#SBATCH --error=logs/quant_baseline_%A_%a.err

set -euo pipefail

: "${RGCLS_REPO_ROOT:=${SLURM_SUBMIT_DIR:?submit from the repository root}}"
: "${RGCLS_PYTHON_ENV:=$RGCLS_REPO_ROOT/.venv}"
CONFIG="${METHOD_CONFIG:-$RGCLS_REPO_ROOT/configs/method_recovery_4h200.example.yaml}"
EXPORT_ROOT="${QB_EXPORT_ROOT:-$RGCLS_REPO_ROOT/outputs/quantization_baselines}"
MODE="${QB_MODE:?QB_MODE must be smoke or formal}"
MODEL_ID="${QB_MODEL_ID:-8b_15}"
if [[ "$MODEL_ID" != 8b_15 && "$MODEL_ID" != 8b_25 ]]; then
  echo "QB_MODEL_ID must be 8b_15 or 8b_25" >&2
  exit 2
fi

mkdir -p "$RGCLS_REPO_ROOT/logs" "$EXPORT_ROOT/artifacts"
mapfile -t MATRIX <<'EOF'
basis_sharing|Basis Sharing|BF16
basis_sharing|Basis Sharing|W8A16
basis_sharing|Basis Sharing|W4A16
svd_llm|SVD-LLM|BF16
svd_llm|SVD-LLM|W8A16
svd_llm|SVD-LLM|W4A16
EOF

IFS='|' read -r method_slug method_label precision <<< "${MATRIX[$SLURM_ARRAY_TASK_ID]}"
checkpoint="${QB_CHECKPOINT_ROOT:?set QB_CHECKPOINT_ROOT to the recovered baseline checkpoint tree}/$method_slug/$MODEL_ID/seed_44/best_model"
output="$EXPORT_ROOT/$MODE/$method_slug/${precision,,}"
artifact="$EXPORT_ROOT/artifacts/${method_slug}_${MODEL_ID}_cekd_s44_${precision,,}.pt"
smoke_args=()
if [[ "$MODE" == smoke ]]; then
  smoke_args=(--max-total-examples 64)
fi

export PYTHONPATH="$RGCLS_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$RGCLS_REPO_ROOT/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

module load miniconda3/26.1.1
srun --nodes=1 --ntasks=1 --gpus-per-task=4 --cpus-per-task=48 \
  "$RGCLS_PYTHON_ENV/bin/python" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$RGCLS_REPO_ROOT/reproduction/baselines/quantization/quantized_eval.py" \
  --config "$CONFIG" \
  --checkpoint "$checkpoint" \
  --method "$method_label" \
  --model-id "$MODEL_ID" \
  --precision "$precision" \
  --seed 44 \
  --artifact "$artifact" \
  --output "$output" \
  --evaluation-data-root "${QB_EVALUATION_DATA_ROOT:?set QB_EVALUATION_DATA_ROOT}" \
  "${smoke_args[@]}"
