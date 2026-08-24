#!/usr/bin/env python3
"""Validate the small, version-controlled artifacts in this research release."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]

EXPECTED_HASHES = {
    "catalog_v1.json": "a0d822c081987f318ea52cf4cfb3d0ee073ddd1560d9253b754ee383a126ff9a",
    "discoveries_v1.json": "f985ca912e9d2be32eb125eb6dcd9015a1b97341a8f9355755b3a854019545f3",
    "paper/physics_gnn_symbolic_discovery.pdf": "7552e711f624f71a17bbc1c41d3eacba0d0af6b2248b84aa6401ef810109c261",
}


def load_json(path: str):
    with (ROOT / path).open(encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_graph() -> None:
    graph = load_json("cm_fullgraph_v1.json")
    nodes = graph["node_metadata"]
    edges = graph["edge_metadata"]
    node_types: dict[str, int] = {}
    for node in nodes:
        node_types[node["node_type"]] = node_types.get(node["node_type"], 0) + 1

    require(len(nodes) == 305, f"expected 305 graph nodes, found {len(nodes)}")
    require(len(edges) == 1217, f"expected 1,217 graph edges, found {len(edges)}")
    require(
        node_types == {"quantity": 159, "equation": 131, "law": 15},
        f"unexpected node-type counts: {node_types}",
    )


def validate_history() -> None:
    history = load_json("v9_history.json")
    require(len(history) == 42, f"expected 42 archived epochs, found {len(history)}")

    best_epoch = min(history, key=lambda row: row["val_mae"])
    epoch_40 = next(row for row in history if row["ep"] == 40)
    require(best_epoch["ep"] == 37, f"expected best MAE at epoch 37, found epoch {best_epoch['ep']}")
    require(math.isclose(best_epoch["val_mae"], 0.02356, abs_tol=5e-6), "best MAE drift")

    expected_ladder = {
        "L0": 0.98996,
        "L1": 0.99401,
        "L2": 0.98605,
        "L3": 0.99139,
        "L4": 0.98595,
    }
    for key, expected in expected_ladder.items():
        require(
            math.isclose(epoch_40[key], expected, abs_tol=5e-6),
            f"epoch-40 {key} drift: expected {expected}, found {epoch_40[key]}",
        )


def validate_discoveries() -> None:
    discoveries = load_json("discoveries_v1.json")
    catalog = load_json("catalog_v1.json")
    all_scores = discoveries["all_scores"]

    require(len(all_scores) == 400, f"expected 400 screened pairs, found {len(all_scores)}")
    require(len(catalog) == 400, f"expected 400 catalog rows, found {len(catalog)}")

    score_pairs = {(row["A"], row["B"]) for row in all_scores}
    catalog_pairs = {(row["A"], row["B"]) for row in catalog}
    require(len(score_pairs) == 400, "screen contains duplicate quantity pairs")
    require(score_pairs == catalog_pairs, "screen and audit catalog quantity pairs differ")

    verdict_counts: dict[str, int] = {}
    for row in catalog:
        verdict = row["verdict"]
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
    expected_counts = {
        "PROVEN PURE": 12,
        "PROVEN CONDITIONAL": 242,
        "BAYESIAN ARTIFACT": 33,
        "ADDITIVE-MEDIATED": 91,
        "UNRESOLVED-INVARIANT": 6,
        "UNRESOLVED": 16,
    }
    require(verdict_counts == expected_counts, f"unexpected audit counts: {verdict_counts}")


def validate_hashes() -> None:
    for relative_path, expected in EXPECTED_HASHES.items():
        actual = sha256(ROOT / relative_path)
        require(actual == expected, f"SHA-256 mismatch for {relative_path}: {actual}")


def main() -> int:
    checks = [validate_graph, validate_history, validate_discoveries, validate_hashes]
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    print("Release validation passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, StopIteration, ValueError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
