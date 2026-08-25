# Changelog

All notable research-package changes are documented here. Published artifacts are never replaced in place.

## 1.1.0 - 2026-08-25

### Added

- Rewrote and expanded the manuscript around the author's graph-based view of
  physics, including a multi-route mass example and explicit tables for the
  32-dimensional node features and 16-dimensional edge features.
- Added a deterministic inference-time intervention study for SI node
  dimensions, semantic node features, edge operators, edge role/direction,
  edge flags, and reverse edges.
- Added a near-parameter-matched flat MLP baseline with three independent
  seeds, validation-only checkpoint selection, a target-mean reference, and an
  analytic formula oracle.
- Added an exact CPU dependency lock, a reproducibility map, a real-data
  roadmap, a responsible-publicity kit, and an isolated multi-seed GNN runner.

### Changed

- The manuscript now uses the stable concept DOI, which always resolves to the
  newest archival version.
- Clarified the distinction between the archived one-seed GNN result,
  inference-time checkpoint diagnostics, and independently trained multi-seed
  baselines.
- Added the author's Bihar background and long-term AI-for-physics mission as
  motivation while explicitly keeping new-physics discovery outside the
  current paper's claims.

## 1.0.1 - 2026-08-24

### Corrected

- Corrected the epoch associated with the best archived validation magnitude MAE. The value 0.02356 occurs at epoch 37 in `v9_history.json`, not epoch 31 as stated in version 1.0.0.
- Rebuilt and visually verified the four-page manuscript PDF. The metric value, epoch-40 validation MAE, sign accuracy, held-out route results, and scientific conclusions are unchanged.
- Updated stale text that described the Zenodo archive as a future step.

### Added

- Automated validation of graph counts, archived training metrics, the 400-pair audit, and release checksums.
- Automatic paper builds on manuscript changes.
- DOI, citation, CodeMeta, Zenodo, contribution, and issue-reporting metadata.

## 1.0.0 - 2026-08-18

- Initial public research release with code, manuscript, 400-pair audit catalog, large experiment artifacts, GitHub release, and Zenodo DOI.
