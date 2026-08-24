# Physics GNN Symbolic Discovery

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21984786.svg)](https://doi.org/10.5281/zenodo.21984786)
[![Validate release artifacts](https://github.com/mi99at/physics-gnn-symbolic-discovery/actions/workflows/validate-release.yml/badge.svg)](https://github.com/mi99at/physics-gnn-symbolic-discovery/actions/workflows/validate-release.yml)
[![Build paper PDF](https://github.com/mi99at/physics-gnn-symbolic-discovery/actions/workflows/build-paper.yml/badge.svg)](https://github.com/mi99at/physics-gnn-symbolic-discovery/actions/workflows/build-paper.yml)
[![License: MIT](https://img.shields.io/badge/code%20license-MIT-blue.svg)](LICENSE)
[![Data and paper: CC BY 4.0](https://img.shields.io/badge/data%20%26%20paper-CC%20BY%204.0-green.svg)](LICENSE-DATA.md)

This repository contains the code for a self-auditing graph neural network for a closed, synthetic classical-mechanics corpus. The goal is deliberately modest and testable: learn to infer missing physical quantities from partial observations, probe the trained solver for candidate relationships, and then audit those candidates against the equations used to generate the corpus.

The project does **not** claim to discover new physics. In a closed corpus, every valid relationship must be a consequence of the encoded equations. A useful discovery pipeline should identify that fact and distinguish it from distribution-dependent correlations.

**Read the paper:** [PDF](paper/physics_gnn_symbolic_discovery.pdf) · [DOI record](https://doi.org/10.5281/zenodo.21984786) · [Versioned release](https://github.com/mi99at/physics-gnn-symbolic-discovery/releases/tag/v1.0.0)

## Results at a glance

| Component | Archived result |
|---|---:|
| Typed physics graph | 305 nodes, 1,217 directed edges |
| Scenario-consistent training corpus | 650,000 worlds across 13 domains |
| Held-out route tests at epoch 40 | $R^2=0.986$--$0.994$ |
| Candidate relationships screened and audited | 400 |
| Proven monomial consequences | 254 |
| Prior-sensitive artifacts | 33 |

These counts are machine-checked by `scripts/validate_release.py`. The remaining catalog entries are explicitly marked as additive-mediated, unresolved-invariant, or unresolved rather than being presented as discoveries.

## What is in this release

- `cm_fullgraph_builder.py` constructs the typed physics graph.
- `scenario_generator_v1.py` generates internally consistent physical worlds and verifies representative identities.
- `gnn_trainer_v9.py` trains the operator-conditioned graph attention network.
- `discover_v1.py` counterfactually probes the trained model for distant quantity pairs.
- `derivation_checker_v1.py` certifies monomial relations with exact log-space elimination and tests prior sensitivity.
- `cm_fullgraph_v1.json`, `scenario_meta_v1.json`, and `v9_history.json` are small, versioned experiment artifacts.

Large data arrays and PyTorch checkpoints are intentionally not tracked in Git. They should be attached to the tagged release through Zenodo or GitHub Releases, with their checksums recorded in `ARTIFACTS.md`.

## Verified v9 experiment snapshot

The saved graph contains 305 nodes (159 quantities, 131 equations, and 15 laws) and 1,217 edges. The training corpus contains 650,000 scenario-consistent worlds across 13 domains. At epoch 40, the archived run reports a validation magnitude MAE of 0.02386 and the following held-out, banned-combination test R² values:

| Test | L0 | L1 | L2 | L3 | L4 |
|---|---:|---:|---:|---:|---:|
| R² at epoch 40 | 0.990 | 0.994 | 0.986 | 0.991 | 0.986 |

The five tests are defined in `scenario_meta_v1.json`. They hold out routes such as `(frequency, radius) → tangential velocity` and `(wavelength, frequency, linear density) → string tension`.

## Reproducing the core experiment

Use Python 3.10+ and install the dependencies:

```bash
python -m pip install -r requirements.txt
python cm_fullgraph_builder.py
python scenario_generator_v1.py 50000
python gnn_trainer_v9.py
python discover_v1.py
python derivation_checker_v1.py
```

The full training run is GPU-oriented. `python gnn_trainer_v9.py --smoke` is a short sanity check; it is not intended to reproduce the reported final results.

To verify the archived graph, training metrics, 400-pair audit, and release checksums without retraining:

```bash
python scripts/validate_release.py
```

Every push and pull request runs the same release validation automatically. Changes to the manuscript also trigger a clean LaTeX build.

## Publication status

The complete 400-candidate discovery screen and matching catalog are tracked in this repository. The large scenario arrays and checkpoints are attached to the [v1.0.0 release](https://github.com/mi99at/physics-gnn-symbolic-discovery/releases/tag/v1.0.0), with checksums in `ARTIFACTS.md`. See `RESULTS_STATUS.md` for the audit boundary and verified results.

The manuscript is available as a [compiled PDF](paper/physics_gnn_symbolic_discovery.pdf) and as [LaTeX source](paper/physics_gnn_symbolic_discovery.tex). The published archival record is available at DOI [10.5281/zenodo.21984786](https://doi.org/10.5281/zenodo.21984786).

## Citation

Please cite the software metadata in `CITATION.cff`:

> Md Minnatullah (2026). *Physics GNN Symbolic Discovery: A Self-Auditing Graph Neural Network for Classical Mechanics*. Zenodo. https://doi.org/10.5281/zenodo.21984786

Machine-readable citation and software metadata are provided in `CITATION.cff`, `codemeta.json`, and `.zenodo.json`.

## Contributing

Reproduction reports, bug reports, and carefully scoped extensions are welcome. Please read `CONTRIBUTING.md` before opening an issue or pull request.

## License

Source code is available under the MIT License. The paper, generated data, model checkpoints, and other research artifacts are available under CC BY 4.0; see `LICENSE-DATA.md`.
