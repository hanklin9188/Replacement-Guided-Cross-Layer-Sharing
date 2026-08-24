# External baseline reproduction slots

The paper compares against Basis Sharing and SVD-LLM, but those methods are run
on a separate server. Each reserved method directory has four stable locations:

- `vendor/`: pinned upstream source snapshot or Git submodule metadata;
- `configs/`: paper-matched compression/recovery/evaluation configurations;
- `code/`: thin adapters that emit the shared payload contract;
- `slurm/`: smoke, full, serialization, and quantization launchers.

Do not commit model weights, datasets, private paths, credentials, or scheduler
logs. Export results into `data/external/<method>/incoming/` using the schemas in
`data/external/schema/`.
