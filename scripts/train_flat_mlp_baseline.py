#!/usr/bin/env python3
"""Train reproducible non-graph baselines for the held-out ladder exam.

The learned baseline receives exactly the revealed normalized magnitudes,
revealed signs, and reveal mask.  It receives no graph, node features, edge
features, equation identities, law identities, or scenario label.  Training
uses the same 30--80% reveal rule and the same banned-combination protection as
the archived GNN.  The script also reports a target-mean baseline and an
analytic formula oracle as lower and upper reference points.

This is intentionally a baseline study, not a replacement for retraining the
archived GNN across independent seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import platform
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = pathlib.Path(__file__).resolve().parents[1]
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29])
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-size", type=int, default=10000)
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Optional deterministic training-row limit for a quick probe.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "experiments" / "v1_1_flat_mlp_baseline.json",
    )
    return parser.parse_args()


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FlatMLP(nn.Module):
    """A near-parameter-matched predictor with no relational structure."""

    def __init__(self, n_quantities: int, hidden: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(3 * n_quantities, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.mag_head = nn.Linear(hidden, n_quantities)
        self.sign_head = nn.Linear(hidden, 2 * n_quantities)
        self.n_quantities = n_quantities

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.mag_head(h), self.sign_head(h).view(-1, self.n_quantities, 2)


@dataclass(frozen=True)
class Arrays:
    mags: np.ndarray
    signs: np.ndarray
    exist: np.ndarray
    test_mags: np.ndarray
    test_signs: np.ndarray
    test_reveal: np.ndarray
    test_target: np.ndarray
    test_ban: np.ndarray


def load_arrays() -> tuple[dict, Arrays]:
    meta = json.loads((ROOT / "scenario_meta_v1.json").read_text(encoding="utf-8"))
    train = np.load(ROOT / "scenario_data_v1.npz")
    test = np.load(ROOT / "scenario_test_v1.npz")
    arrays = Arrays(
        mags=train["mags"],
        signs=train["signs"],
        exist=train["exist"],
        test_mags=test["mags"],
        test_signs=test["signs"],
        test_reveal=test["reveal"],
        test_target=test["target"],
        test_ban=test["ban_id"],
    )
    return meta, arrays


def build_bans(meta: dict) -> tuple[np.ndarray, list[tuple[np.ndarray, int]]]:
    qcol = {name: idx for idx, name in enumerate(meta["qty_order"])}
    const = np.zeros(len(qcol), dtype=bool)
    for col in meta["constant_cols"].values():
        const[col] = True
    bans = [
        (np.asarray([qcol[name] for name in ban["inputs"]]), qcol[ban["target"]])
        for ban in meta["bans"]
    ]
    return const, bans


def sample_reveal(
    exist: np.ndarray,
    const: np.ndarray,
    bans: list[tuple[np.ndarray, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Mirror the archived trainer's reveal distribution and ban protection."""
    batch, n_quantities = exist.shape
    fraction = rng.random((batch, 1), dtype=np.float32) * 0.5 + 0.3
    reveal = exist & ((rng.random(exist.shape, dtype=np.float32) < fraction) | const)

    hidden = exist & ~reveal
    for row in np.flatnonzero(hidden.sum(axis=1) == 0):
        candidates = np.flatnonzero(exist[row] & ~const)
        if len(candidates):
            reveal[row, rng.choice(candidates)] = False

    for input_cols, target_col in bans:
        ban_input = np.zeros(n_quantities, dtype=bool)
        ban_input[input_cols] = True
        hidden = exist & ~reveal
        violation = hidden[:, target_col] & (
            (reveal & ~const & ~ban_input).sum(axis=1) == 0
        )
        for row in np.flatnonzero(violation):
            candidates = np.flatnonzero(exist[row] & ~reveal[row] & ~ban_input)
            candidates = candidates[candidates != target_col]
            if len(candidates):
                reveal[row, rng.choice(candidates)] = True
    return reveal


def make_input(mags: np.ndarray, signs: np.ndarray, reveal: np.ndarray) -> np.ndarray:
    reveal32 = reveal.astype(np.float32, copy=False)
    return np.concatenate(
        [mags * reveal32, signs.astype(np.float32, copy=False) * reveal32, reveal32],
        axis=1,
    )


def r2_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    ban_ids: np.ndarray,
    n_bans: int,
) -> dict:
    ladder: dict[str, dict[str, float | int]] = {}
    for ban_id in range(n_bans):
        select = ban_ids == ban_id
        pred = predictions[select].astype(np.float64)
        truth = targets[select].astype(np.float64)
        residual = pred - truth
        ss_total = np.square(truth - truth.mean()).sum()
        ladder[f"L{ban_id}"] = {
            "n": int(select.sum()),
            "r2": float(1.0 - np.square(residual).sum() / max(ss_total, 1e-12)),
            "mae_log10_scaled": float(np.abs(residual).mean()),
            "rmse_log10_scaled": float(np.sqrt(np.square(residual).mean())),
        }
    residual = predictions.astype(np.float64) - targets.astype(np.float64)
    return {
        "ladder": ladder,
        "aggregate": {
            "n": int(len(targets)),
            "mae_log10_scaled": float(np.abs(residual).mean()),
            "rmse_log10_scaled": float(np.sqrt(np.square(residual).mean())),
        },
    }


def predict_target_mean(
    arrays: Arrays, n_quantities: int, training_rows: np.ndarray
) -> np.ndarray:
    means = np.zeros(n_quantities, dtype=np.float64)
    for col in range(n_quantities):
        present = arrays.exist[training_rows, col]
        values = arrays.mags[training_rows, col]
        means[col] = values[present].mean() if present.any() else 0.0
    return means[arrays.test_target]


def denormalize_abs(normalized_log_magnitude: np.ndarray) -> np.ndarray:
    return np.maximum(np.power(10.0, 10.0 * normalized_log_magnitude) - EPS, 0.0)


def normalize_abs(value: np.ndarray) -> np.ndarray:
    return np.log10(np.abs(value) + EPS) / 10.0


def predict_oracle(meta: dict, arrays: Arrays) -> np.ndarray:
    """Compute each registered ladder relation directly in physical space."""
    qcol = {name: idx for idx, name in enumerate(meta["qty_order"])}
    values = denormalize_abs(arrays.test_mags.astype(np.float64))
    prediction = np.empty(len(values), dtype=np.float64)
    formulas = {
        0: lambda v: 2.0 * math.pi * v[:, qcol["Q_r"]] * v[:, qcol["Q_f"]],
        1: lambda v: 2.0 * math.pi * v[:, qcol["Q_r"]] / v[:, qcol["Q_T_period"]],
        2: lambda v: 0.5 * v[:, qcol["Q_m"]]
        * np.square(2.0 * math.pi * v[:, qcol["Q_r"]] / v[:, qcol["Q_T_period"]]),
        3: lambda v: v[:, qcol["Q_A_amp"]]
        * np.sqrt(v[:, qcol["Q_k_spr"]] / v[:, qcol["Q_m"]]),
        4: lambda v: v[:, qcol["Q_mu_lin"]]
        * np.square(v[:, qcol["Q_f"]] * v[:, qcol["Q_lambda"]]),
    }
    for ban_id, formula in formulas.items():
        select = arrays.test_ban == ban_id
        prediction[select] = normalize_abs(formula(values[select]))
    return prediction


@torch.inference_mode()
def predict_mlp(
    model: FlatMLP,
    arrays: Arrays,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    for start in range(0, len(arrays.test_mags), batch_size):
        stop = min(start + batch_size, len(arrays.test_mags))
        x = make_input(
            arrays.test_mags[start:stop],
            arrays.test_signs[start:stop],
            arrays.test_reveal[start:stop],
        )
        magnitude, _ = model(torch.from_numpy(x).to(device))
        target = torch.from_numpy(arrays.test_target[start:stop]).to(device)
        pred = magnitude.gather(1, target[:, None]).squeeze(1)
        predictions.append(pred.cpu().numpy())
    return np.concatenate(predictions)


@torch.inference_mode()
def validation_mae(
    model: FlatMLP,
    arrays: Arrays,
    rows: np.ndarray,
    reveal: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    """Magnitude MAE on a fixed validation subset and fixed reveal mask."""
    model.eval()
    absolute_error = 0.0
    hidden_count = 0
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        batch_rows = rows[start:stop]
        batch_reveal = reveal[start:stop]
        hidden = arrays.exist[batch_rows] & ~batch_reveal
        x = torch.from_numpy(
            make_input(
                arrays.mags[batch_rows],
                arrays.signs[batch_rows],
                batch_reveal,
            )
        ).to(device)
        target = torch.from_numpy(arrays.mags[batch_rows]).to(device)
        hidden_t = torch.from_numpy(hidden).to(device)
        magnitude, _ = model(x)
        absolute_error += float((magnitude[hidden_t] - target[hidden_t]).abs().sum())
        hidden_count += int(hidden.sum())
    return absolute_error / max(hidden_count, 1)


def train_one_seed(
    seed: int,
    args: argparse.Namespace,
    arrays: Arrays,
    const: np.ndarray,
    bans: list[tuple[np.ndarray, int]],
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    validation_reveal: np.ndarray,
    device: torch.device,
) -> tuple[dict, int]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_quantities = arrays.mags.shape[1]
    model = FlatMLP(n_quantities, args.hidden).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    history: list[dict] = []
    started = time.perf_counter()
    best_validation_mae = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(train_rows)
        total_loss = total_mae = total_sign_accuracy = 0.0
        steps = 0
        epoch_started = time.perf_counter()
        for start in range(0, len(order), args.batch_size):
            rows = order[start : start + args.batch_size]
            exist = arrays.exist[rows]
            reveal = sample_reveal(exist, const, bans, rng)
            hidden = exist & ~reveal
            x = torch.from_numpy(
                make_input(arrays.mags[rows], arrays.signs[rows], reveal)
            ).to(device)
            target_magnitude = torch.from_numpy(arrays.mags[rows]).to(device)
            target_sign = torch.from_numpy(arrays.signs[rows].astype(np.int64)).to(device)
            hidden_t = torch.from_numpy(hidden).to(device)

            magnitude, sign_logits = model(x)
            magnitude_loss = F.huber_loss(magnitude[hidden_t], target_magnitude[hidden_t])
            signed_hidden = hidden_t & (target_sign != 0)
            sign_class = (target_sign[signed_hidden] > 0).long()
            sign_loss = F.cross_entropy(sign_logits[signed_hidden], sign_class)
            loss = magnitude_loss + 0.2 * sign_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                total_loss += float(loss)
                total_mae += float((magnitude[hidden_t] - target_magnitude[hidden_t]).abs().mean())
                total_sign_accuracy += float(
                    (sign_logits[signed_hidden].argmax(-1) == sign_class).float().mean()
                )
                steps += 1
        val_mae = validation_mae(
            model,
            arrays,
            validation_rows,
            validation_reveal,
            device,
            args.batch_size,
        )
        if val_mae < best_validation_mae:
            best_validation_mae = val_mae
            best_epoch = epoch + 1
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / steps,
            "train_hidden_mae_log10_scaled": total_mae / steps,
            "train_hidden_sign_accuracy": total_sign_accuracy / steps,
            "validation_hidden_mae_log10_scaled": val_mae,
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        print(
            f"seed={seed} epoch={epoch + 1}/{args.epochs} "
            f"loss={row['train_loss']:.4f} val_mae={val_mae:.5f} "
            f"({row['elapsed_seconds']:.1f}s)",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("no MLP checkpoint was selected")
    model.load_state_dict(best_state)
    prediction = predict_mlp(model, arrays, device, args.batch_size)
    target = arrays.test_mags[np.arange(len(arrays.test_target)), arrays.test_target]
    result = r2_metrics(prediction, target, arrays.test_ban, len(bans))
    result.update(
        {
            "seed": seed,
            "epochs": args.epochs,
            "selected_epoch": best_epoch,
            "selected_validation_hidden_mae_log10_scaled": best_validation_mae,
            "test_evaluated_once_after_validation_selection": True,
            "elapsed_seconds": time.perf_counter() - started,
            "history": history,
        }
    )
    return result, parameters


def summarize_seeds(seed_results: list[dict]) -> dict:
    summary: dict[str, dict[str, float]] = {}
    for ladder in seed_results[0]["ladder"]:
        values = np.asarray([row["ladder"][ladder]["r2"] for row in seed_results])
        summary[ladder] = {
            "r2_mean": float(values.mean()),
            "r2_std_population": float(values.std(ddof=0)),
            "r2_min": float(values.min()),
            "r2_max": float(values.max()),
        }
    mae = np.asarray([row["aggregate"]["mae_log10_scaled"] for row in seed_results])
    return {
        "ladder": summary,
        "aggregate_mae_log10_scaled_mean": float(mae.mean()),
        "aggregate_mae_log10_scaled_std_population": float(mae.std(ddof=0)),
    }


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta, arrays = load_arrays()
    const, bans = build_bans(meta)
    split_permutation = np.random.default_rng(1).permutation(len(arrays.mags))
    validation_count = int(len(arrays.mags) * 0.05)
    validation_pool = split_permutation[:validation_count]
    all_rows = split_permutation[validation_count:]
    validation_rng = np.random.default_rng(20260825)
    validation_rows = np.sort(
        validation_rng.choice(
            validation_pool,
            min(args.validation_size, len(validation_pool)),
            replace=False,
        )
    )
    validation_reveal = sample_reveal(
        arrays.exist[validation_rows],
        const,
        bans,
        np.random.default_rng(20260825),
    )
    if args.train_limit is not None:
        limit_rng = np.random.default_rng(20260825)
        all_rows = np.sort(limit_rng.choice(all_rows, args.train_limit, replace=False))

    target = arrays.test_mags[np.arange(len(arrays.test_target)), arrays.test_target]
    references = {
        "target_mean": r2_metrics(
            predict_target_mean(arrays, arrays.mags.shape[1], all_rows),
            target,
            arrays.test_ban,
            len(bans),
        ),
        "analytic_formula_oracle": r2_metrics(
            predict_oracle(meta, arrays), target, arrays.test_ban, len(bans)
        ),
    }

    output = {
        "study_type": "multi-seed flat MLP baseline on the fixed ladder exam",
        "claim_boundary": (
            "The learned baseline has no graph or structural features. The archived GNN "
            "was not retrained here, so this is not a multi-seed GNN comparison."
        ),
        "data": {
            "training_archive": "scenario_data_v1.npz",
            "training_archive_sha256": sha256(ROOT / "scenario_data_v1.npz"),
            "test_archive": "scenario_test_v1.npz",
            "test_archive_sha256": sha256(ROOT / "scenario_test_v1.npz"),
            "training_rows_used": int(len(all_rows)),
            "validation_rows_excluded": validation_count,
            "validation_rows_evaluated": int(len(validation_rows)),
            "validation_mask_seed": 20260825,
            "split": "same fixed NumPy RNG seed-1 95/5 split as gnn_trainer_v9.py",
            "test_rows": int(len(arrays.test_mags)),
        },
        "environment": {
            "device": str(device),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "configuration": {
            "seeds": args.seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden": args.hidden,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "checkpoint_selection": "lowest fixed-mask validation magnitude MAE",
            "test_policy": "ladder exam evaluated once after checkpoint selection",
            "input": "revealed magnitude + revealed sign + reveal mask (3 x 159)",
            "excluded": "graph, node features, edge features, equations, laws, scenario id",
            "masking": meta["train_sampling_rule"],
        },
        "reference_baselines": references,
        "seeds": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(output, indent=2) + "\n").encode("utf-8"))

    parameter_count = 0
    for seed in args.seeds:
        result, parameter_count = train_one_seed(
            seed,
            args,
            arrays,
            const,
            bans,
            all_rows,
            validation_rows,
            validation_reveal,
            device,
        )
        output["seeds"].append(result)
        output["model_parameters"] = parameter_count
        output["summary"] = summarize_seeds(output["seeds"])
        args.output.write_bytes((json.dumps(output, indent=2) + "\n").encode("utf-8"))

    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
