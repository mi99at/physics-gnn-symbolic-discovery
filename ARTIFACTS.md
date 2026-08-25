# Release artifacts

Large arrays and checkpoints are released separately from ordinary Git
tracking. Small reports, the paper, and their validation hashes are tracked.

| Artifact | Purpose | Release location | SHA-256 |
|---|---|---|---|
| scenario_data_v1.npz | 650,000 generated worlds | GitHub release v1.0.0 | 38696d30f86a3aa84696502de89f2ea7befbc5af1f1767f1dfb8d066372b4ec4 |
| scenario_test_v1.npz | 7,500 held-out ladder worlds | GitHub release v1.0.0 | 87771caaef98d3ede91f34e74fb56fca6df3078408b2582bd5b8a3ec3c1c2ba4 |
| v9_best.pt | best archived checkpoint | GitHub release v1.0.0 | 66c3cf45b1476b40e3e65a84ea6106b51ca911a665d9bf145bb46314eec7302e |
| v9_latest.pt | latest archived checkpoint | GitHub release v1.0.0 | 7d1b6b647d0ee59d5991a62f5568b90c57a031be2aec1333da66dd077005cbd9 |
| paper/physics_gnn_symbolic_discovery.pdf | expanded, visually verified v1.1 manuscript | tracked in Git | c7245a0c19787250eac2c455e0d7373b941ef9ac220112958df50a5e75318995 |
| experiments/v1_1_checkpoint_ablations.json | deterministic inference-time feature interventions | tracked in Git | 5ad9298ef70db5541829a6934b33aa20a86a255b6a923fa79943bf01b77beff1 |
| experiments/v1_1_flat_mlp_baseline.json | three-seed validation-selected non-graph baseline | tracked in Git | a2fdbc91d27a4566e19064eca71e5c5bad4a504284545252e6dc91e3e8242efb |
| assets/physics-graph-social-preview.png | publicity illustration, not experimental evidence | tracked in Git | 15c872c6cc78f7d340f43b4291bc278b3ce24514547c98cdd69a514fa0f4e035 |
| physics_gnn_symbolic_discovery_v1.1.0.zip | compact paper, diagnostics, documentation, and metadata bundle | v1.1.0 release attachment | 7b10dd348425396fd93ab8f9cbbc2fcdae9f78357817c75f3d9491b2652d57f7 |
| discoveries_v1.json | 400-pair neural candidate screen | tracked in Git | f985ca912e9d2be32eb125eb6dcd9015a1b97341a8f9355755b3a854019545f3 |
| catalog_v1.json | matching audit of all 400 candidates | tracked in Git | a0d822c081987f318ea52cf4cfb3d0ee073ddd1560d9253b754ee383a126ff9a |

Large-artifact release:
https://github.com/mi99at/physics-gnn-symbolic-discovery/releases/tag/v1.0.0

The stable archival concept DOI is
[10.5281/zenodo.21984785](https://doi.org/10.5281/zenodo.21984785). Published
files are never replaced in place; corrections and expanded manuscripts are
released as new versions.
