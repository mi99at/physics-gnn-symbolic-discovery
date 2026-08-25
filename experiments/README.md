# Experiments

- `v1_1_checkpoint_ablations.json`: inference-time interventions on the single
  archived `v9_best.pt` checkpoint, evaluated on a deterministic 100-example
  subset of each ladder route.
- `v1_1_flat_mlp_baseline.json`: three-seed learned baseline with no graph or
  structural features, plus a target-mean reference and an analytic formula
  oracle.

Every experiment file declares its own claim boundary. In particular, zeroing
features at inference time does not replace retraining an ablated model, and a
multi-seed baseline does not make the archived one-seed GNN a multi-seed result.
