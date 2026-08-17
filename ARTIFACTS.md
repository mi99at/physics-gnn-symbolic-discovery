# Release artifacts

The following files are required for a complete, reproducible release but are too large for ordinary Git tracking:

| Artifact | Purpose | Release location | SHA-256 |
|---|---|---|---|
| `scenario_data_v1.npz` | 650,000 generated worlds | pending | pending |
| `scenario_test_v1.npz` | held-out ladder test set | pending | pending |
| `v9_best.pt` | best archived checkpoint | pending | pending |
| `v9_latest.pt` | latest archived checkpoint | pending | pending |
| final `discoveries_v1.json` | 400-pair discovery screen | tracked in Git | `f985ca912e9d2be32eb125eb6dcd9015a1b97341a8f9355755b3a854019545f3` |
| final `catalog_v1.json` | audit results for all candidates | tracked in Git | `a0d822c081987f318ea52cf4cfb3d0ee073ddd1560d9253b754ee383a126ff9a` |

For publication, upload these files to a versioned Zenodo record (recommended) and link the record from the GitHub release. Do not replace a published artifact in place; create a new release version instead.
