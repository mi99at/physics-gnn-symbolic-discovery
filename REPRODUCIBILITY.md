# Reproducibility map

This repository separates archived evidence from new diagnostics so that later
experiments cannot silently change the claims made by the original run.

## Evidence classes

1. **Archived training run.** `v9_history.json`, `v9_best.pt`, and
   `v9_latest.pt` come from the original operator-conditioned GNN run. The
   reported epoch-40 ladder scores and epoch-37 best validation MAE are read
   from that immutable history. The checkpoint SHA-256 is recorded in
   `ARTIFACTS.md`.
2. **Checkpoint interventions.** `experiments/v1_1_checkpoint_ablations.json`
   evaluates one archived checkpoint after zeroing selected node or edge
   features, or removing reverse edges. These are inference-time diagnostic
   interventions; they are not independently retrained ablations.
3. **Non-graph baselines.** `experiments/v1_1_flat_mlp_baseline.json` trains a
   flat multilayer perceptron across three independent seeds. It receives
   revealed magnitudes, signs, and the reveal mask, but no graph, node features,
   edge features, equations, laws, or scenario label.
4. **Discovery audit.** `discoveries_v1.json` is the neural candidate screen;
   `catalog_v1.json` is the matching 400-pair symbolic and prior-sensitivity
   audit. Candidate relationships are not counted as new physical laws.

## Environment

The v1.1 CPU diagnostics were run with Python 3.12, NumPy 2.3.5, and PyTorch
2.13.0. Install the exact tested environment with:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-lock.txt
```

On Linux or macOS, replace `.venv/Scripts/python` with `.venv/bin/python`.
Full GNN training is GPU-oriented; use an appropriate CUDA PyTorch wheel while
preserving the NumPy version and record the substituted package versions.

## Verification commands

Validate the tracked graph, archived metrics, audit counts, diagnostics, and
checksums:

```bash
python scripts/validate_release.py
```

Re-run the inference-time feature interventions against the archived
checkpoint:

```bash
python scripts/evaluate_checkpoint_ablations.py \
  --per-ban 100 --batch-size 8 \
  --output experiments/v1_1_checkpoint_ablations.json
```

Re-run the non-graph baseline study:

```bash
python scripts/train_flat_mlp_baseline.py \
  --epochs 5 --seeds 7 17 29 --batch-size 2048 --hidden 768 \
  --output experiments/v1_1_flat_mlp_baseline.json
```

The JSON outputs include data and checkpoint hashes, package versions, device,
configuration, per-seed histories, and all reported metrics.

The next full GNN comparison is pre-wired to use isolated checkpoint
directories so it cannot overwrite the archived run:

    python scripts/run_gnn_multiseed.py --seeds 7 17 29 43 71 --epochs 60

This command is GPU-oriented. Its outputs belong under
experiments/gnn_multiseed/ and should be published only after all requested
seeds complete.

## Claim boundary

The current corpus is synthetic and generated from the encoded equations. It
can establish route learning, auditability, and sensitivity to graph features;
it cannot establish a new law of nature. Real measurements, independent
replication, noise models, and retrained multi-seed GNN comparisons remain the
required next stage.
