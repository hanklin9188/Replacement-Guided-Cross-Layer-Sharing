# Audit notes

## Numeric authorities

1. `data/processed/paper_main_table.csv` is the frozen paper aggregate.
2. `data/ours/per_task.csv` is the compact current Ours task-level export used
   to recompute run macros, seed means, and sample SDs.
3. `data/processed/paired_bootstrap_results.csv` is based on synchronized
   current per-example evaluation across all methods.
4. `data/ours/quantization.csv` contains Ours quantization observations only.

The frozen and synchronized layers are deliberately distinct. Maximum observed
Ours rerun drift is 0.341 percentage point; therefore the bootstrap artifact
must not be presented as if it exactly regenerated every frozen headline cell.

## Enforced checks

`scripts/verify_repository.py` checks:

- 294 Ours task rows -> 42 run macros -> 18 aggregate rows;
- sample SD (`ddof=1`) for every CE and CE+KD operating point;
- 1,748 directed rows, 874 undirected pairs, 11 non-singleton groups;
- `mutual_cost=max(direction)` and `delta <= Delta` identities;
- six observed joint-distortion rows and five ablation rows;
- 288 paired-bootstrap rows and exact 252-row question coverage;
- 18 standalone byte-accounting rows;
- bibliography keys, local Markdown links, private paths, and credential
  markers.

## Scope corrections in the public release

- The parallel adapter rank is recorded as 128, matching the completed paper
  runs and the released recovery contract.
- Unique-parameter reduction and standalone serialized-byte reduction are kept
  in separate columns.
- Sharing does not reduce logical depth or FFN execution count; no inherent
  FLOP-saving claim is made.
- The paper does not silently substitute an Ours-only or fabricated
  cross-method quantization figure when external manifests are absent.

## Code provenance

`src/icassp27/` is a modular reference implementation suitable for inspection
and clean smoke tests. `reproduction/ours/` is the larger paper-era pipeline
snapshot and exact paper recovery wrapper. The two are not represented as
bit-identical implementations; the paper-era snapshot is the relevant path for
full reproduction.
