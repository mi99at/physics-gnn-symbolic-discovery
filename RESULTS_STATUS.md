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

SHA-256 checksums:

- `catalog_v1.json`: `a0d822c081987f318ea52cf4cfb3d0ee073ddd1560d9253b754ee383a126ff9a`
- `discoveries_v1.json`: `f985ca912e9d2be32eb125eb6dcd9015a1b97341a8f9355755b3a854019545f3`

The large scenario arrays and archived checkpoints are attached to the versioned GitHub `v1.0.0` release, with their SHA-256 checksums recorded in `ARTIFACTS.md`. The manuscript and release metadata are archived at DOI [`10.5281/zenodo.21984786`](https://doi.org/10.5281/zenodo.21984786).
