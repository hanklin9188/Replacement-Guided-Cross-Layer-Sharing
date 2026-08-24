# Basis Sharing reproduction record

Upstream: <https://github.com/TUDa-HWAI/Basis_Sharing>

The controlled Llama-3 port is implemented by
`src/icassp27/controlled_baselines/`. It preserves the cross-layer shared-basis
factorization while using the common paper recovery/evaluation contract. The
exact upstream revision is in `vendor/REVISION`; portable configs and Slurm
entry points are at the repository-level baseline paths documented in
`../README.md`.
