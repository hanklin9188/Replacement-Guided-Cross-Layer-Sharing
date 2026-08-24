# Quantization figure: complete data

Figure: `fig_quantization_15_25_storage_accuracy`

## Experimental scope

- Backbone: Llama-3.1-8B
- Recovery: CE+KD
- Seed: 44
- Structural operating points: 15% and 25%
- Precision labels: BF16, INT8 (`W8A16` in raw manifests), INT4 (`W4A16` in raw manifests)
- Evaluator: seven zero-shot multiple-choice tasks, candidate log-likelihood, `length_norm=none`
- Macro: unweighted mean of the seven task accuracies
- Quantized accuracy: deterministic packed weights are unpacked/dequantized for BF16 computation; these results do not measure integer-kernel latency

## Every point plotted in the figure

| Method | Structural reduction | Precision | Macro | Serialized size (GiB) |
|---|---:|---|---:|---:|
| Dense teacher | 0% | BF16 | 0.868500 | 14.960000 |
| Dense teacher | 0% | INT8 | 0.869200 | 8.460000 |
| Dense teacher | 0% | INT4 | 0.693300 | 5.410000 |
| FAD (Ours) | 15% | BF16 | 0.864800 | 12.670000 |
| FAD (Ours) | 15% | INT8 | 0.865200 | 7.320000 |
| FAD (Ours) | 15% | INT4 | 0.839100 | 4.720000 |
| Basis Sharing | 15% | BF16 | 0.771615 | 12.714103 |
| Basis Sharing | 15% | INT8 | 0.772831 | 7.339127 |
| Basis Sharing | 15% | INT4 | 0.366489 | 4.791077 |
| SVD-LLM | 15% | BF16 | 0.765499 | 12.714116 |
| SVD-LLM | 15% | INT8 | 0.765538 | 7.339349 |
| SVD-LLM | 15% | INT4 | 0.646174 | 4.772380 |
| FAD (Ours) | 25% | BF16 | 0.862800 | 11.040000 |
| FAD (Ours) | 25% | INT8 | 0.863400 | 6.500000 |
| FAD (Ours) | 25% | INT4 | 0.787600 | 4.390000 |
| Basis Sharing | 25% | BF16 | 0.705600 | 11.218345 |
| Basis Sharing | 25% | INT8 | 0.705800 | 6.591168 |
| Basis Sharing | 25% | INT4 | 0.359297 | 4.363619 |
| SVD-LLM | 25% | BF16 | 0.704607 | 11.218362 |
| SVD-LLM | 25% | INT8 | 0.701196 | 6.591368 |
| SVD-LLM | 25% | INT4 | 0.576959 | 4.374444 |

Dense-teacher and FAD sizes above are the reported rounded GiB values. Basis Sharing and SVD-LLM sizes are converted from the exact artifact byte counts below using 1 GiB = 2^30 bytes.

## Quantization differences and paired confidence intervals

Differences are relative to the BF16 checkpoint of the same method and structural operating point.

| Method | Structural reduction | Comparison | Macro difference (pp) | Paired 95% CI (pp) |
|---|---:|---|---:|---:|
| Dense teacher | 0% | INT8 - BF16 | +0.070 | unavailable |
| Dense teacher | 0% | INT4 - BF16 | -17.520 | unavailable |
| FAD (Ours) | 15% | INT8 - BF16 | +0.041 | [-0.105, +0.198] |
| FAD (Ours) | 15% | INT4 - BF16 | -2.572 | [-3.123, -2.019] |
| Basis Sharing | 15% | INT8 - BF16 | +0.122 | [-0.110, +0.356] |
| Basis Sharing | 15% | INT4 - BF16 | -40.513 | [-41.777, -39.265] |
| SVD-LLM | 15% | INT8 - BF16 | +0.004 | [-0.258, +0.269] |
| SVD-LLM | 15% | INT4 - BF16 | -11.932 | [-12.881, -10.986] |
| FAD (Ours) | 25% | INT8 - BF16 | +0.064 | [-0.043, +0.171] |
| FAD (Ours) | 25% | INT4 - BF16 | -7.519 | [-8.271, -6.772] |
| Basis Sharing | 25% | INT8 - BF16 | +0.020 | [-0.213, +0.252] |
| Basis Sharing | 25% | INT4 - BF16 | -34.630 | [-35.900, -33.352] |
| SVD-LLM | 25% | INT8 - BF16 | -0.341 | [-0.646, -0.046] |
| SVD-LLM | 25% | INT4 - BF16 | -12.765 | [-13.774, -11.753] |

Dense-teacher differences are derived from rounded aggregate Macro values. Example-level dense-teacher predictions were not supplied, so no paired confidence interval is available.

## Exact serialized artifacts for Basis Sharing and SVD-LLM

| Method | Structural reduction | Precision | Artifact bytes | GiB | Reduction vs method BF16 | Reduction vs dense BF16 teacher |
|---|---:|---|---:|---:|---:|---:|
| Basis Sharing | 15% | BF16 | 13,651,664,196 | 12.714103 | 0.000% | 14.999% |
| Basis Sharing | 15% | INT8 | 7,880,327,115 | 7.339127 | 42.276% | 50.934% |
| Basis Sharing | 15% | INT4 | 5,144,379,319 | 4.791077 | 62.317% | 67.969% |
| SVD-LLM | 15% | BF16 | 13,651,678,274 | 12.714116 | 0.000% | 14.999% |
| SVD-LLM | 15% | INT8 | 7,880,565,485 | 7.339349 | 42.274% | 50.932% |
| SVD-LLM | 15% | INT4 | 5,124,303,477 | 4.772380 | 62.464% | 68.094% |
| Basis Sharing | 25% | BF16 | 12,045,606,387 | 11.218345 | 0.000% | 24.999% |
| Basis Sharing | 25% | INT8 | 7,077,212,958 | 6.591168 | 41.247% | 55.934% |
| Basis Sharing | 25% | INT4 | 4,685,400,434 | 4.363619 | 61.103% | 70.827% |
| SVD-LLM | 25% | BF16 | 12,045,624,693 | 11.218362 | 0.000% | 24.999% |
| SVD-LLM | 25% | INT8 | 7,077,427,700 | 6.591368 | 41.245% | 55.933% |
| SVD-LLM | 25% | INT4 | 4,697,023,860 | 4.374444 | 61.006% | 70.754% |

The dense BF16 denominator stored in the baseline manifests is 16,060,643,284 bytes.

## Seven-task accuracy: 15% structural reduction

### FAD (Ours)

| Task | BF16 | INT8 | INT4 |
|---|---:|---:|---:|
| ARC-Challenge | 0.7918 | 0.7884 | 0.7534 |
| ARC-Easy | 0.9011 | 0.9011 | 0.8817 |
| HellaSwag | 0.9526 | 0.9530 | 0.9325 |
| OpenBookQA | 0.8720 | 0.8780 | 0.8320 |
| PIQA | 0.8743 | 0.8721 | 0.8509 |
| Social-IQA | 0.7968 | 0.7989 | 0.7810 |
| WinoGrande | 0.8650 | 0.8650 | 0.8421 |
| Macro | 0.8648 | 0.8652 | 0.8391 |

### Basis Sharing

| Task | BF16 | INT8 | INT4 |
|---|---:|---:|---:|
| ARC-Challenge | 0.648464 | 0.648464 | 0.255973 |
| ARC-Easy | 0.793350 | 0.794613 | 0.299242 |
| HellaSwag | 0.876319 | 0.877415 | 0.312388 |
| OpenBookQA | 0.734000 | 0.738000 | 0.284000 |
| PIQA | 0.783460 | 0.782372 | 0.516866 |
| Social-IQA | 0.773286 | 0.770215 | 0.389458 |
| WinoGrande | 0.792423 | 0.798737 | 0.507498 |
| Macro | 0.771615 | 0.772831 | 0.366489 |

### SVD-LLM

| Task | BF16 | INT8 | INT4 |
|---|---:|---:|---:|
| ARC-Challenge | 0.644198 | 0.645904 | 0.500000 |
| ARC-Easy | 0.797559 | 0.793771 | 0.678030 |
| HellaSwag | 0.869050 | 0.868253 | 0.667397 |
| OpenBookQA | 0.732000 | 0.738000 | 0.612000 |
| PIQA | 0.774755 | 0.775299 | 0.706746 |
| Social-IQA | 0.756397 | 0.755374 | 0.675537 |
| WinoGrande | 0.784530 | 0.782163 | 0.683504 |
| Macro | 0.765499 | 0.765538 | 0.646174 |

## Seven-task accuracy: 25% structural reduction

### FAD (Ours)

| Task | BF16 | INT8 | INT4 |
|---|---:|---:|---:|
| ARC-Challenge | 0.7986 | 0.8003 | 0.6681 |
| ARC-Easy | 0.8918 | 0.8944 | 0.8152 |
| HellaSwag | 0.9451 | 0.9448 | 0.8782 |
| OpenBookQA | 0.8620 | 0.8620 | 0.7380 |
| PIQA | 0.8711 | 0.8716 | 0.8237 |
| Social-IQA | 0.7989 | 0.7989 | 0.7677 |
| WinoGrande | 0.8721 | 0.8721 | 0.8224 |
| Macro | 0.8628 | 0.8634 | 0.7876 |

### Basis Sharing

| Task | BF16 | INT8 | INT4 |
|---|---:|---:|---:|
| ARC-Challenge | 0.554608 | 0.558020 | 0.249147 |
| ARC-Easy | 0.723485 | 0.720960 | 0.284933 |
| HellaSwag | 0.819060 | 0.818064 | 0.257518 |
| OpenBookQA | 0.676000 | 0.678000 | 0.284000 |
| PIQA | 0.734494 | 0.732318 | 0.497280 |
| Social-IQA | 0.716479 | 0.722108 | 0.445752 |
| WinoGrande | 0.715075 | 0.711129 | 0.496448 |
| Macro | 0.705600 | 0.705800 | 0.359297 |

### SVD-LLM

| Task | BF16 | INT8 | INT4 |
|---|---:|---:|---:|
| ARC-Challenge | 0.555461 | 0.555461 | 0.425768 |
| ARC-Easy | 0.712121 | 0.709175 | 0.600168 |
| HellaSwag | 0.791376 | 0.790380 | 0.554272 |
| OpenBookQA | 0.676000 | 0.660000 | 0.570000 |
| PIQA | 0.739935 | 0.739391 | 0.658868 |
| Social-IQA | 0.734391 | 0.733367 | 0.646366 |
| WinoGrande | 0.722968 | 0.720600 | 0.583268 |
| Macro | 0.704607 | 0.701196 | 0.576959 |

Per-task dense-teacher predictions were not part of the supplied data; only its aggregate Macro and storage values are used in this figure.

## Source files

- Figure data: `data/processed/quantization/storage_accuracy.csv`
- Baseline per-task accuracy:
  `data/processed/quantization/baseline_{15,25}_accuracy.csv`
- Exact packed-byte manifests:
  `data/processed/quantization/baseline_{15,25}_bytes.csv`
- Paired bootstrap:
  `data/processed/quantization/baseline_{15,25}_paired_bootstrap.csv`
- Evaluation/packing implementation:
  `reproduction/baselines/quantization/quantized_eval.py`
- Figure generator: `scripts/generate_quantization_figure.py`
