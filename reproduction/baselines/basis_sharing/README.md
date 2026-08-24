# Basis Sharing integration slot

Upstream: <https://github.com/TUDa-HWAI/Basis_Sharing>

Required additions from the external server:

1. Pin the exact upstream commit in `vendor/REVISION` and place the source or
   submodule under `vendor/`.
2. Add matched 3B/8B configurations for 15/20/25% under `configs/`.
3. Add adapters under `code/` for compression, CE recovery, CE+KD recovery,
   seven-task evaluation, standalone serialization, and TorchAO quantization.
4. Add dependency-gated `smoke.sbatch`, `full.sbatch`, `serialize.sbatch`, and
   `quantize.sbatch` under `slurm/`.
5. Emit the four files required by `data/external/README.md`.

The common evaluator contract is log-probability multiple-choice scoring with
`length_norm=none`, tasks fixed by the paper, and seeds 42/43/44.
