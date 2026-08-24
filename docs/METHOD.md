# Method

## Directed full-model replacement

Let a frozen decoder contain FFNs `F_1, ..., F_L`. For donor `i` and target
`j`, the model is evaluated after replacing only `F_j` with `F_i`. The directed
cost is

```text
C(i -> j) = DeltaNLL(i -> j) + lambda_KL KL(p_0 || p_(i -> j)),
```

with `lambda_KL = 1`. Because the replacement is performed inside the full
model, `C(i -> j)` need not equal `C(j -> i)`. The released observations contain
756 directed 3B rows and 992 directed 8B rows.

Implementation map:

- measurement: `src/icassp27/replacement.py`;
- paper-era distributed collector:
  `reproduction/ours/core/ffn_functional_redundancy_ddp.py`;
- observed path-free matrix: `data/ours/directed_costs.csv`.

## Fixed-budget grouping

For a candidate group `G`, define the symmetric pair cost and complete-link
group bound

```text
C_bar(i,j) = max(C(i -> j), C(j -> i))
Delta(G)   = max C_bar(i,j), for i != j in G.
```

Starting from singleton layers, the algorithm greedily merges the two groups
with the smallest complete-link cross cost until exactly `K` stored FFNs remain.
This is a one-step minimax greedy rule; it is not claimed to solve the global
partition problem.

Implementation map:

- modular grouping: `src/icassp27/grouping.py`;
- paper-era grouping and policy export:
  `reproduction/ours/core/analyze_ffn_grouping_views.py`;
- observed final groups: `data/ours/group_analysis.csv`.

## Directional representative

Once a group is fixed, its representative is chosen in the deployment
direction:

```text
m*(G) = argmin_m max_j C(m -> j),  m,j in G and j != m.
```

The best representative cost `delta(G)` is therefore bounded by the complete
pair envelope `Delta(G)`. All 11 released non-singleton groups satisfy this
identity, which is enforced by `scripts/verify_repository.py`.

## Shared model and recovery

Each logical layer uses the selected shared FFN plus a layer-private rank-64
parallel low-rank residual adapter. The sharing groups and representatives are
frozen before recovery. Attention remains frozen; the shared FFN bank and
parallel adapters are trainable in the released paper recipe.

Three stages are kept separate:

1. **Pure:** immutable step-0 shared checkpoint, no optimizer update.
2. **CE:** labels only; no teacher is loaded or forwarded.
3. **CE+KD:** decision CE plus token-logit KL to a frozen backbone-matched
   teacher, temperature 2 and KD weight 1.

The exact paper schedule uses 7,500/10,000 optimizer updates for 3B/8B,
maximum length 384, and seeds 42/43/44. See `docs/REPRODUCTION.md` for the
resource and configuration contract.

## Quantization and accounting

Structural compression and post-training quantization are evaluated separately.
W8A16 uses symmetric per-output-channel INT8 weights. W4A16 uses asymmetric
group-128 INT4 in TorchAO's tile-packed TinyGEMM format. Activations remain
BF16; no calibration, QAT, or post-quantization recovery is used.

The repository distinguishes:

- nominal structural target;
- unique-parameter reduction;
- standalone serialized-byte reduction;
- quantized serialized size.

Logical depth and FFN execution count are unchanged, so the method makes a
storage claim rather than an inherent FLOP-reduction claim.
