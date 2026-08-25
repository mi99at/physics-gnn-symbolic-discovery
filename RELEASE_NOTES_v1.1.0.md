# Version 1.1.0 — graph features, controls, and author perspective

This release expands the paper and evidence package without changing the
original archived GNN history.

## Highlights

- Centers the paper on the author's view of physics as a graph of quantities,
  equations, laws, and multiple solution routes.
- Adds a visual multi-route mass example and complete node/edge feature tables.
- Adds a near-parameter-matched flat MLP trained from three independent seeds
  with validation-only checkpoint selection.
- Adds deterministic inference-time interventions on SI and semantic node
  features, edge operators, edge role/direction, edge flags, and reverse edges.
- Adds a reproducibility map, exact CPU dependency lock, isolated multi-seed
  GNN runner, real-data roadmap, and responsible publicity kit.
- Adds the author's Bihar background and long-term AI-for-physics mission while
  explicitly keeping new-physics discovery outside the current claims.

## New controlled result

The flat MLP has 1,916,637 parameters versus 2,165,699 in the archived GNN. Its
mean held-out R² values across seeds 7, 17, and 29 are:

| L0 | L1 | L2 | L3 | L4 |
|---:|---:|---:|---:|---:|
| 0.968 ± 0.006 | 0.925 ± 0.068 | 0.877 ± 0.020 | 0.870 ± 0.056 | 0.515 ± 0.060 |

The archived GNN remains a one-seed result with epoch-40 R² values of 0.990,
0.994, 0.986, 0.991, and 0.986. Seed counts differ, so this is an architectural
reference, not a completed multi-seed GNN comparison.

## Claim boundary

The corpus is synthetic and generated from the encoded equations. The release
tests route learning, sensitivity to designed graph information, and the
ability to audit rediscovered relationships. It does not report a new law of
nature. Inference-time interventions are diagnostic and do not replace
retrained ablations.

## Verification

Run:

    python scripts/validate_release.py

The manifest release_manifest_v1.1.0.json records exact hashes for the paper,
diagnostic reports, scripts, environment lock, and publicity visual.
