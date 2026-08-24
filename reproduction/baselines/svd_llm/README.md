# SVD-LLM reproduction record

Upstream: <https://github.com/AIoT-MLSys-Lab/SVD-LLM>

The controlled Llama-3 port is implemented by
`src/icassp27/controlled_baselines/`. It preserves activation-whitened
truncated SVD and staged U/V recovery while using the common paper evaluator.
The exact upstream revision is in `vendor/REVISION`; portable configs and Slurm
entry points are at the repository-level baseline paths documented in
`../README.md`.
