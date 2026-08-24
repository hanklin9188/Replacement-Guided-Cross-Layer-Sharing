# External baseline import boundary

The public repository already contains the controlled baseline implementation,
compact comparison artifacts, and complete quantization summaries. This tree
defines an optional strict contract for importing full per-example payloads or
reruns from another machine without changing paper-facing code.

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
Imported payloads are validated independently and do not overwrite the frozen
paper tables unless explicitly promoted after audit.
