# Results audit and release status

## Verified from the archived v9 files

- Graph: 305 nodes; 159 quantity nodes, 131 equation nodes, and 15 law nodes; 1,217 edges.
- Dataset: 650,000 training worlds and 7,500 held-out banned-combination test worlds.
- Training history: 42 completed epochs are stored in `v9_history.json`.
- Best stored validation magnitude MAE: 0.02356 (epoch 37).
- Held-out ladder values at epoch 40: L0 0.98996, L1 0.99401, L2 0.98605, L3 0.99139, L4 0.98595.
- The archived discovery file contains a ranked 40-candidate report plus the scores for all 400 screened quantity pairs. The final catalog audits every one of those 400 pairs.

## Final 400-candidate audit

The restored `discoveries_v1.json` and `catalog_v1.json` contain the complete final audit. All 400 candidate pairs match between the two files. The catalog reports 12 proven-pure and 242 proven-conditional records, representing 254 unique proven formulas, alongside 33 prior-sensitive artifacts, 91 additive-mediated cases, 6 unresolved-invariant cases, and 16 unresolved cases.

## Version 1.1 controlled diagnostics

The non-graph baseline has 1,916,637 parameters versus 2,165,699 in the
archived GNN. It receives revealed magnitudes, revealed signs, and the reveal
mask only. It uses the same fixed 95/5 row split, masking distribution, and ban
protection, but no graph, node features, edge features, equation/law nodes, or
scenario identity. Seeds 7, 17, and 29 each train for five epochs; checkpoint
selection uses only magnitude MAE on a fixed 10,000-world validation subset,
and the complete ladder is evaluated once afterward.

Mean ± population standard deviation across the three MLP seeds:

| L0 | L1 | L2 | L3 | L4 |
|---:|---:|---:|---:|---:|
| 0.968 ± 0.006 | 0.925 ± 0.068 | 0.877 ± 0.020 | 0.870 ± 0.056 | 0.515 ± 0.060 |

The archived GNN values remain a one-seed result and are not represented as a
multi-seed comparison.

The checkpoint intervention study uses a deterministic 100-example subset of
each ladder route. Aggregate normalized-log MAE changes from 0.0084 for the
full checkpoint to 0.0521 without SI node dimensions, 0.0538 without semantic
node features, 0.0742 without edge operators, 0.1465 without edge
role/direction, 0.0358 without edge flags, and 0.0683 without reverse edges.
These are inference-time distribution shifts on one checkpoint, not retrained
ablations.

SHA-256 checksums:

- `catalog_v1.json`: `a0d822c081987f318ea52cf4cfb3d0ee073ddd1560d9253b754ee383a126ff9a`
- `discoveries_v1.json`: `f985ca912e9d2be32eb125eb6dcd9015a1b97341a8f9355755b3a854019545f3`

The large scenario arrays and archived checkpoints are attached to the versioned GitHub `v1.0.0` release, with their SHA-256 checksums recorded in `ARTIFACTS.md`. The stable concept DOI [`10.5281/zenodo.21984785`](https://doi.org/10.5281/zenodo.21984785) resolves to the newest archival version.
