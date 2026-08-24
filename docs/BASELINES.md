# Basis Sharing and SVD-LLM reproduction

The repository contains the controlled Llama-3 ports used for the paper. Both
methods share one data loader, CE/CE+KD objective, checkpoint selector,
seven-task evaluator, parameter accountant, and scheduler contract so that the
compression representation is the principal method-specific difference.

## Included implementation

- `src/icassp27/controlled_baselines/compress.py`: activation-whitened
  shared-basis and truncated-SVD compression at measured 15/20/25% budgets.
- `src/icassp27/controlled_baselines/method_recovery.py`: final Basis full
  recovery and staged SVD U-then-V recovery on four H200 GPUs.
- `src/icassp27/controlled_baselines/train_eval.py`: Pure, CE, CE+KD, held-out
  decision-CE selection, and seven-task evaluation.
- `reproduction/baselines/quantization/`: packed INT8/INT4 evaluation,
  aggregation, paired bootstrap, byte accounting, and artifact validation.
- `configs/controlled_baselines.example.yaml` and
  `configs/method_recovery_4h200.example.yaml`: complete portable contracts.

Upstream provenance is pinned at:

- Basis Sharing: `1c021b6ce1d3932750b8cb9f7f0e08fa9acdf2c2`
- SVD-LLM: `7538cca98880ff312a79a16252aab5a7b480fbe9`

The upstream repositories are not copied into this artifact. Clone them under
`third_party/Basis_Sharing` and `third_party/SVD-LLM` when running the provenance
preflight; the executable controlled ports themselves are already included.

## Required external artifacts

Copy each example config to an untracked filename and fill the model, teacher,
fixed recovery split, evaluation-data, and calibration paths under `artifacts/`.
Meta Llama weights, benchmark corpora, teachers, recovered checkpoints, and
multi-GiB packed files are intentionally not redistributed.

## Compression and Pure matrix

Dry-run the complete dependency graph first:

```bash
python reproduction/baselines/scripts/submit_controlled_baselines.py \
  --config configs/controlled_baselines.example.yaml
```

After the preflight succeeds, add `--submit`. Each compression/Pure task uses
one H200. The final paper recovery path uses the already materialized raw
checkpoints and four H200 GPUs:

```bash
python reproduction/baselines/scripts/submit_method_recovery_4h200.py \
  --config configs/method_recovery_4h200.example.yaml
```

The second command is also a dry-run unless its `--submit` flag is supplied.

## Quantization

For each structural point set `QB_MODEL_ID` to `8b_15` or `8b_25`, provide the
recovered checkpoint root and seven-task data root, then submit the six-element
precision/method array:

```bash
export QB_MODE=formal
export QB_MODEL_ID=8b_15
export QB_CHECKPOINT_ROOT=/path/to/recovered/checkpoints
export QB_EVALUATION_DATA_ROOT=/path/to/seven_task_jsonl
sbatch --array=0-5 reproduction/baselines/slurm/quantize_4xh200.sbatch
```

Run `aggregate_quantization.py` and `validate_quantization.py` against the
result tree before promoting compact rows into `data/processed/quantization/`.
The released compact tables cover all 15%/25% method/precision cells.

## Optional raw-payload import

`data/external/` remains available for importing full per-example prediction
payloads from another machine. Those large payloads are not required to build
the paper or regenerate any released figure.
