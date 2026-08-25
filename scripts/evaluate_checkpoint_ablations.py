#!/usr/bin/env python3
"""Run reproducible inference-time feature ablations on the archived v9 model.

This is a diagnostic of one archived training run, not a replacement for
multi-seed retraining.  Each condition uses the same deterministic subset of
the five held-out ladder tests so differences are attributable to the
inference-time graph intervention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import sys
import time
from typing import Iterable

import numpy as np
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gnn_trainer_v9 as gt  # noqa: E402  (project root added above)


CONDITIONS = (
    "full",
    "node_si_zero",
    "node_semantic_zero",
    "edge_operators_zero",
    "edge_role_direction_zero",
    "edge_flags_zero",
    "reverse_edges_removed",
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_graph(condition: str, original: dict[str, torch.Tensor | int]) -> None:
    """Apply one graph intervention to the globals used by PhysicsGNN."""

    gt.Xg = original["Xg"].clone()
    gt.EIg = original["EIg"].clone()
    gt.EFg = original["EFg"].clone()
    gt.EDGES = int(original["EDGES"])

    if condition == "node_si_zero":
        gt.Xg[:, :7] = 0
    elif condition == "node_semantic_zero":
        gt.Xg[:, 7:] = 0
    elif condition == "edge_operators_zero":
        gt.EFg[:, :7] = 0
    elif condition == "edge_role_direction_zero":
        gt.EFg[:, 7:12] = 0
    elif condition == "edge_flags_zero":
        gt.EFg[:, 12:] = 0
    elif condition == "reverse_edges_removed":
        metadata = gt.G["edge_metadata"]
        keep_np = np.array([row["kind"] != "reverse" for row in metadata])
        keep = torch.tensor(keep_np, device=gt.DEV)
        gt.EIg = gt.EIg[:, keep]
        gt.EFg = gt.EFg[keep]
        gt.EDGES = int(keep.sum().item())
    elif condition != "full":
        raise ValueError(f"unknown condition: {condition}")

    gt.EI_FULL = gt.batch_graph(gt.C.batch)
    gt.EF_FULL = gt.EFg.repeat(gt.C.batch, 1)


def choose_rows(ban_id: int, per_ban: int, seed: int) -> np.ndarray:
    rows = np.where(gt.T_BID == ban_id)[0]
    if per_ban >= len(rows):
        return rows
    rng = np.random.default_rng(seed + 1009 * ban_id)
    return np.sort(rng.choice(rows, per_ban, replace=False))


@torch.inference_mode()
def evaluate_condition(
    model: torch.nn.Module,
    *,
    per_ban: int,
    batch_size: int,
    subset_seed: int,
) -> dict[str, object]:
    model.eval()
    ladder: dict[str, dict[str, float | int]] = {}
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for ban_id in range(len(gt.BANS)):
        selected = choose_rows(ban_id, per_ban, subset_seed)
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []

        for offset in range(0, len(selected), batch_size):
            rows = selected[offset : offset + batch_size]
            batch = len(rows)
            mag_q = torch.tensor(gt.T_MAGS[rows], device=gt.DEV)
            sign_q = torch.tensor(gt.T_SIGNS[rows], device=gt.DEV, dtype=torch.float32)
            reveal = torch.tensor(gt.T_REVEAL[rows], device=gt.DEV)

            mag_n = torch.zeros(batch, gt.NODES, device=gt.DEV)
            sign_n = torch.zeros(batch, gt.NODES, device=gt.DEV)
            known_n = torch.zeros(batch, gt.NODES, device=gt.DEV)
            qnodes = gt.QNODE.unsqueeze(0).expand(batch, -1)
            mag_n.scatter_(1, qnodes, mag_q * reveal)
            sign_n.scatter_(1, qnodes, sign_q * reveal)
            known_n.scatter_(1, qnodes, reveal.float())

            edge_index = gt.batch_graph(batch)
            predicted_nodes, _ = model(mag_n, sign_n, known_n, batch, edge_index)
            predicted_q = predicted_nodes.view(batch, gt.NODES).gather(1, qnodes)
            target_col = torch.tensor(gt.T_TGT[rows], device=gt.DEV).unsqueeze(1)
            predictions.append(
                predicted_q.gather(1, target_col).squeeze(1).float().cpu().numpy()
            )
            targets.append(mag_q.gather(1, target_col).squeeze(1).cpu().numpy())

        pred = np.concatenate(predictions)
        true = np.concatenate(targets)
        residual = pred - true
        denominator = float(np.square(true - true.mean()).sum())
        r2 = 1.0 - float(np.square(residual).sum()) / max(denominator, 1e-12)
        ladder[f"L{ban_id}"] = {
            "n": int(len(true)),
            "r2": r2,
            "mae_log10_scaled": float(np.abs(residual).mean()),
            "rmse_log10_scaled": float(np.sqrt(np.square(residual).mean())),
        }
        all_predictions.append(pred)
        all_targets.append(true)

    pred_all = np.concatenate(all_predictions)
    true_all = np.concatenate(all_targets)
    residual_all = pred_all - true_all
    return {
        "ladder": ladder,
        "aggregate": {
            "n": int(len(true_all)),
            "mae_log10_scaled": float(np.abs(residual_all).mean()),
            "rmse_log10_scaled": float(np.sqrt(np.square(residual_all).mean())),
        },
    }


def parse_conditions(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(items) - set(CONDITIONS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown condition(s): {', '.join(unknown)}")
    if not items:
        raise argparse.ArgumentTypeError("at least one condition is required")
    return items


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=ROOT / "v9_best.pt")
    parser.add_argument("--per-ban", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--subset-seed", type=int, default=20260825)
    parser.add_argument(
        "--conditions",
        type=parse_conditions,
        default=list(CONDITIONS),
        help="comma-separated conditions (default: all)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "experiments" / "v1_1_checkpoint_ablations.json",
    )
    args = parser.parse_args(argv)

    if args.per_ban <= 0 or args.batch_size <= 0:
        parser.error("--per-ban and --batch-size must be positive")

    checkpoint = args.checkpoint.resolve()
    state = torch.load(checkpoint, map_location=gt.DEV, weights_only=False)
    model = gt.PhysicsGNN().to(gt.DEV)
    model.load_state_dict(state["model"])

    original: dict[str, torch.Tensor | int] = {
        "Xg": gt.Xg.clone(),
        "EIg": gt.EIg.clone(),
        "EFg": gt.EFg.clone(),
        "EDGES": gt.EDGES,
    }
    report: dict[str, object] = {
        "study_type": "inference-time feature ablation of one archived checkpoint",
        "claim_boundary": (
            "Diagnostic only: these interventions do not replace independent retraining "
            "across multiple seeds."
        ),
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_epoch": int(state["epoch"]),
        "archived_best_validation_mae": float(state["best"]),
        "device": gt.DEV,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "subset": {
            "seed": args.subset_seed,
            "per_ban": args.per_ban,
            "batch_size": args.batch_size,
        },
        "conditions": {},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for condition in args.conditions:
        configure_graph(condition, original)
        started = time.perf_counter()
        result = evaluate_condition(
            model,
            per_ban=args.per_ban,
            batch_size=args.batch_size,
            subset_seed=args.subset_seed,
        )
        result["elapsed_seconds"] = time.perf_counter() - started
        report["conditions"][condition] = result
        args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
        ladder_text = " ".join(
            f"{name}={metrics['r2']:+.4f}"
            for name, metrics in result["ladder"].items()
        )
        print(f"{condition:24s} {ladder_text} ({result['elapsed_seconds']:.1f}s)")

    configure_graph("full", original)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
