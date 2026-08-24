# Limitations

- **External baseline completeness.** Processed Basis Sharing and SVD-LLM
  comparison artifacts are present, but their full source/config/checkpoint and
  quantization manifests still need to be imported from another server. The
  repository labels this status throughout and withholds the final cross-method
  quantization figure until validation passes.
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
- **Quantized runtime.** The reported TorchAO results establish storage and
  accuracy behavior under a common weight-only setup. Kernel and hardware
  effects are configuration-dependent and should not be generalized to every
  serving stack.
- **Backbone and task scope.** Conclusions are limited to Llama-3.2-3B,
  Llama-3.1-8B, the stated 15/20/25% operating points, and seven
  multiple-choice tasks.
