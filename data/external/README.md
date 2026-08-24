# External baseline import boundary

This machine contains the Ours implementation and compact processed comparison
artifacts. The complete Basis Sharing and SVD-LLM implementations, checkpoints,
per-example predictions, byte manifests, and quantization manifests are produced
on another server. This tree reserves every destination and defines one strict
contract so those assets can be added without changing the paper-facing code.

For each method, place these files in its `incoming/` directory:

| File | Required content |
|---|---|
| `predictions.csv` | 6 operating points x 3 seeds x 7 tasks, one row per example |
| `byte_manifest.csv` | 2 backbones x 3 compression targets |
| `quantization.csv` | 8B 15/25% x BF16/W8A16/W4A16 |
| `run_manifest.json` | upstream revision, model revisions, evaluator, environment, and commands |

Templates live in `schema/`. After transfer, submit the scheduled import chain:

```bash
sbatch slurm/import_external_baselines.sbatch
```

The job validates schemas and exact coverage before it writes processed tables.
The quantization figure generator refuses to create the manuscript-facing file
until both methods have complete inputs.
