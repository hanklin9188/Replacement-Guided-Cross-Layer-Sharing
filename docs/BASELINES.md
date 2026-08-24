# Basis Sharing and SVD-LLM handoff

## Why the slots are empty

The current machine hosts Ours. Full Basis Sharing and SVD-LLM runs are owned
by another server, so this repository includes processed comparison outputs and
strict integration contracts without pretending their full implementation is
already local.

## Required method package

Populate each method under `reproduction/baselines/<method>/` with:

- a pinned upstream source snapshot or revision record in `vendor/`;
- six matched compression/recovery configurations in `configs/`;
- adapters for compression, CE, CE+KD, evaluation, serialization, and
  TorchAO W8A16/W4A16 in `code/`;
- dependency-gated smoke, full, serialization, and quantization jobs in
  `slurm/`.

The upstream repositories are:

- Basis Sharing: <https://github.com/TUDa-HWAI/Basis_Sharing>
- SVD-LLM: <https://github.com/AIoT-MLSys-Lab/SVD-LLM>

## Required result payload

For each method, copy the following into
`data/external/<method>/incoming/`:

1. `predictions.csv`
2. `byte_manifest.csv`
3. `quantization.csv`
4. `run_manifest.json`

Schemas are in `data/external/schema/`. Predictions must cover six model
targets, seeds 42/43/44, and all seven tasks with stable source indices.
Quantization must cover 8B--15/25% at BF16, W8A16, and W4A16 under the common
TorchAO contract.

## Scheduled import

Set `OURS_PREDICTIONS` to the full Ours per-example file, then submit:

```bash
export OURS_PREDICTIONS=/path/to/ours_predictions.csv
bash slurm/submit_external_chain.sh
```

The chain is:

```text
schema/hash validation
  -> 10,000-sample task-stratified paired bootstrap + quantization figure
    -> public repository audit
```

A failed stage blocks every downstream stage. Diagnose its Slurm log before
resubmitting.
