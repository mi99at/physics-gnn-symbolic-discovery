# Physics GNN Symbolic Discovery

This repository contains the code for a self-auditing graph neural network for a closed, synthetic classical-mechanics corpus. The goal is deliberately modest and testable: learn to infer missing physical quantities from partial observations, probe the trained solver for candidate relationships, and then audit those candidates against the equations used to generate the corpus.

The project does **not** claim to discover new physics. In a closed corpus, every valid relationship must be a consequence of the encoded equations. A useful discovery pipeline should identify that fact and distinguish it from distribution-dependent correlations.

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

## Publication status

The complete 400-candidate discovery screen and matching catalog are tracked in this repository. The large scenario arrays and checkpoints are attached to the [v1.0.0 release](https://github.com/mi99at/physics-gnn-symbolic-discovery/releases/tag/v1.0.0), with checksums in `ARTIFACTS.md`. See `RESULTS_STATUS.md` for the audit boundary and verified results.

The manuscript is available as a [compiled PDF](paper/physics_gnn_symbolic_discovery.pdf) and as [LaTeX source](paper/physics_gnn_symbolic_discovery.tex). Its reserved archival DOI is [10.5281/zenodo.21984786](https://doi.org/10.5281/zenodo.21984786); the link will resolve after the Zenodo record is published.

## Citation

Please cite the software metadata in `CITATION.cff`:

> Md Minnatullah (2026). *Physics GNN Symbolic Discovery: A Self-Auditing Graph Neural Network for Classical Mechanics*. Zenodo. https://doi.org/10.5281/zenodo.21984786

## License

Source code is available under the MIT License. The paper, generated data, model checkpoints, and other research artifacts are available under CC BY 4.0; see `LICENSE-DATA.md`.
