# Release artifacts

The following files are required for a complete, reproducible release but are too large for ordinary Git tracking:

| Artifact | Purpose | Release location | SHA-256 |
|---|---|---|---|
| `scenario_data_v1.npz` | 650,000 generated worlds | GitHub release `v1.0.0` | `38696d30f86a3aa84696502de89f2ea7befbc5af1f1767f1dfb8d066372b4ec4` |
| `scenario_test_v1.npz` | held-out ladder test set | GitHub release `v1.0.0` | `87771caaef98d3ede91f34e74fb56fca6df3078408b2582bd5b8a3ec3c1c2ba4` |
| `v9_best.pt` | best archived checkpoint | GitHub release `v1.0.0` | `66c3cf45b1476b40e3e65a84ea6106b51ca911a665d9bf145bb46314eec7302e` |
| `v9_latest.pt` | latest archived checkpoint | GitHub release `v1.0.0` | `7d1b6b647d0ee59d5991a62f5568b90c57a031be2aec1333da66dd077005cbd9` |
| final `discoveries_v1.json` | 400-pair discovery screen | tracked in Git | `f985ca912e9d2be32eb125eb6dcd9015a1b97341a8f9355755b3a854019545f3` |
| final `catalog_v1.json` | audit results for all candidates | tracked in Git | `a0d822c081987f318ea52cf4cfb3d0ee073ddd1560d9253b754ee383a126ff9a` |

Release page: https://github.com/mi99at/physics-gnn-symbolic-discovery/releases/tag/v1.0.0

For long-term archival, the same files can later be mirrored to a versioned Zenodo record and linked from this release. Do not replace a published artifact in place; create a new release version instead.
