# Controlled baseline reproduction

This tree contains the executable Basis Sharing and SVD-LLM paper pipeline.
The shared implementation is in `src/icassp27/controlled_baselines/`; this
directory holds orchestration, quantization, scheduler jobs, and upstream
revision records.

- `scripts/submit_controlled_baselines.py`: compression, Pure, and the original
  controlled CE/CE+KD matrix (1xH200 per task).
- `scripts/submit_method_recovery_4h200.py`: the final paper-matched recovery
  matrix (4xH200 per task).
- `quantization/`: deterministic packed INT8/INT4 evaluation, aggregation, and
  artifact validation.
- `slurm/`: portable NCHC launchers; paths are supplied through environment
  variables rather than embedded private locations.

The separately licensed upstream repositories are not vendored. Exact commits
are recorded in each method's `vendor/REVISION`; clone them only when the
provenance preflight is required.
