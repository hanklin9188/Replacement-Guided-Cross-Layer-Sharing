# Results and provenance

## Frozen paper aggregate

The paper-facing table is frozen in `data/processed/paper_main_table.csv`.
Scores are seven-task macro accuracy percentages; CE and CE+KD use seeds
42/43/44.

| Backbone | Target | Method | Pure | CE | CE+KD |
|---|---:|---|---:|---:|---:|
| 3B | 15% | Basis Sharing | 33.20 | 71.04 ± 0.12 | 71.28 ± 0.52 |
| 3B | 15% | SVD-LLM | 33.20 | 57.36 ± 20.32 | 70.78 ± 0.13 |
| 3B | 15% | Ours | 33.21 | **85.35 ± 0.17** | **84.83 ± 0.47** |
| 3B | 20% | Basis Sharing | 33.20 | 48.91 ± 13.84 | 68.15 ± 0.54 |
| 3B | 20% | SVD-LLM | 33.20 | 33.49 ± 0.22 | 67.32 ± 0.28 |
| 3B | 20% | Ours | 33.95 | **85.07 ± 0.54** | **85.06 ± 0.08** |
| 3B | 25% | Basis Sharing | 33.22 | 33.73 ± 0.48 | 65.06 ± 0.20 |
| 3B | 25% | SVD-LLM | 33.22 | 33.62 ± 0.37 | 63.58 ± 0.73 |
| 3B | 25% | Ours | 33.77 | **84.60 ± 0.12** | **84.09 ± 0.18** |
| 8B | 15% | Basis Sharing | 33.27 | 76.05 ± 0.39 | 76.43 ± 0.47 |
| 8B | 15% | SVD-LLM | 33.21 | 74.43 ± 0.19 | 75.31 ± 0.89 |
| 8B | 15% | Ours | 36.27 | **85.54 ± 0.33** | **86.14 ± 0.33** |
| 8B | 20% | Basis Sharing | 33.20 | **72.81 ± 0.75** | 73.89 ± 0.54 |
| 8B | 20% | SVD-LLM | 33.20 | 58.77 ± 21.74 | 73.01 ± 0.74 |
| 8B | 20% | Ours | 33.65 | 68.04 ± 17.02 | **85.94 ± 0.34** |
| 8B | 25% | Basis Sharing | 33.20 | 33.16 ± 0.04 | 69.96 ± 0.64 |
| 8B | 25% | SVD-LLM | 33.21 | 33.32 ± 0.26 | 69.64 ± 0.34 |
| 8B | 25% | Ours | 34.05 | **81.67 ± 5.96** | **85.92 ± 0.36** |

## Structural evidence

- Median directed asymmetry is 0.221 on 3B and 0.140 on 8B; the 90th
  percentiles are 7.31 and 6.24.
- Every released group satisfies `delta(G) <= Delta(G)`; the maximum observed
  envelope gap is 0.569.
- Per-layer joint cost `C* = C_joint / L` rises from 0.619 to 1.193 on 3B and
  from 0.682 to 1.314 on 8B as compression increases.
- At 3B--20%, removing directionality reduces CE+KD from 85.12% to 83.29% and
  raises `C*` from 1.099 to 1.216. A symmetric representative reaches 80.06%
  and `C*=3.132`.

The exact rows are in `data/ours/pair_analysis.csv`,
`data/ours/group_analysis.csv`, `data/ours/joint_analysis.csv`, and
`data/ours/structural_ablation.csv`.

## Quantized checkpoints

| Structure | Precision | Macro | Size | Reduction vs dense BF16 |
|---|---|---:|---:|---:|
| 8B--15% | BF16 | 86.48% | 12.67 GiB | 15.26% |
| 8B--15% | W8A16 | 86.52% | 7.32 GiB | 51.07% |
| 8B--15% | W4A16 | 83.91% | 4.72 GiB | 68.44% |
| 8B--25% | BF16 | 86.28% | 11.04 GiB | 26.17% |
| 8B--25% | W8A16 | 86.34% | 6.50 GiB | 56.53% |
| 8B--25% | W4A16 | 78.76% | 4.39 GiB | 70.63% |

The full cross-method points, per-task baseline accuracy, paired intervals, and
exact packed-byte manifests are in `data/processed/quantization/`; the complete
numeric report is in `docs/QUANTIZATION.md`.

## Provenance split

The frozen paper table remains the headline authority. The paired-bootstrap
analysis uses synchronized current per-example predictions for all methods and
is stored in `data/processed/paired_bootstrap_results.csv`. Rerun drift from the
frozen aggregates is at most 0.341 percentage point and is not silently mixed
back into the headline table. See `docs/AUDIT.md`.
