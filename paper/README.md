# Paper source

physics_gnn_symbolic_discovery.tex is the source for the current manuscript;
physics_gnn_symbolic_discovery.pdf is the compiled and visually verified
version. The paper uses the stable concept DOI
[10.5281/zenodo.21984785](https://doi.org/10.5281/zenodo.21984785), which
always resolves to the newest archival version.

## Version 1.1

The expanded manuscript:

- presents the author's view of physical worlds as graphs of nodes and edges;
- uses several valid paths to mass as the motivating example;
- documents the complete node-feature and edge-feature encodings;
- adds a three-seed, near-parameter-matched non-graph baseline;
- adds diagnostic interventions on SI/semantic node features, operator/role
  edge features, flags, and reverse edges;
- explains the author's Bihar background, resource constraints, and long-term
  mission without presenting the synthetic benchmark as new physics; and
- gives a concrete route from the closed benchmark to real measurements.

The baseline is selected by validation data only. The feature interventions
operate on one archived checkpoint at inference time and are explicitly not
described as retrained ablations.

The large data arrays and checkpoints are attached to the versioned release
listed in ../ARTIFACTS.md.
