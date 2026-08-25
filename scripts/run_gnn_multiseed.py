#!/usr/bin/env python3
"""Launch independent v9 GNN training directories for a multi-seed study.

This runner does not alter or overwrite the archived v9 checkpoint. Each seed
gets a separate directory containing checkpoints, history, and a console log.
Full runs are GPU-oriented.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29, 43, 71])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=ROOT / "experiments" / "gnn_multiseed",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the trainer's short CPU sanity mode for every seed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        seed_dir = (args.output_root / f"seed_{seed}").resolve()
        seed_dir.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["PHYSICS_GNN_SEED"] = str(seed)
        environment["PHYSICS_GNN_EPOCHS"] = str(args.epochs)
        environment["PHYSICS_GNN_CKDIR"] = str(seed_dir)
        command = [sys.executable, str(ROOT / "gnn_trainer_v9.py")]
        if args.smoke:
            command.append("--smoke")
        log_path = seed_dir / "train.log"
        print(f"seed={seed} -> {seed_dir}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode:
            print(f"seed={seed} failed; see {log_path}", file=sys.stderr)
            return completed.returncode
    print("all requested seeds completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
