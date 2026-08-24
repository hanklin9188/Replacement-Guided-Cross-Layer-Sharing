# Limitations

- **External artifact boundary.** Controlled Basis Sharing and SVD-LLM code,
  configs, launchers, and compact formal outputs are included. Separately
  licensed model weights, teachers, benchmark corpora, recovered checkpoints,
  and multi-GiB packed files are not redistributed.
- **Frozen versus synchronized evaluation.** The headline table is the frozen
  paper aggregate, while paired bootstrap uses synchronized current
  per-example predictions. Maximum Ours drift is 0.341 percentage point. These
  sources are useful for different questions and are not silently merged.
- **Reference versus paper-era implementation.** `src/icassp27/` is a clean
  modular reference implementation. Full paper reproduction uses the larger
  snapshot under `reproduction/ours/`; the two are not claimed to be
  bit-identical.
- **Weights and datasets.** A clean checkout cannot reproduce full GPU results
  without separately licensed Meta Llama weights, benchmark data, teachers,
  and fixed split artifacts.
- **Storage, not inherent compute reduction.** Cross-layer sharing preserves
  logical depth and FFN execution count. Lower parameter and serialized-byte
  counts do not by themselves imply lower FLOPs or latency.
- **Pairwise versus joint behavior.** The group envelope controls measured
  isolated replacements, but it is not a numerical upper bound on simultaneous
  deployment distortion.
- **Quantized runtime.** The released baseline packed-weight evaluator
  dequantizes to BF16 for inference; its results establish storage and accuracy,
  not integer-kernel latency or throughput. Kernel and hardware effects should
  not be generalized to every serving stack.
- **Backbone and task scope.** Conclusions are limited to Llama-3.2-3B,
  Llama-3.1-8B, the stated 15/20/25% operating points, and seven
  multiple-choice tasks.
