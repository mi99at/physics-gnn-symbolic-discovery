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
    "paper/physics_gnn_symbolic_discovery.pdf": "c7245a0c19787250eac2c455e0d7373b941ef9ac220112958df50a5e75318995",
    "experiments/v1_1_checkpoint_ablations.json": "5ad9298ef70db5541829a6934b33aa20a86a255b6a923fa79943bf01b77beff1",
    "experiments/v1_1_flat_mlp_baseline.json": "a2fdbc91d27a4566e19064eca71e5c5bad4a504284545252e6dc91e3e8242efb",
    "assets/physics-graph-social-preview.png": "15c872c6cc78f7d340f43b4291bc278b3ce24514547c98cdd69a514fa0f4e035",
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


def validate_checkpoint_diagnostics() -> None:
    report = load_json("experiments/v1_1_checkpoint_ablations.json")
    require(
        report["study_type"]
        == "inference-time feature ablation of one archived checkpoint",
        "unexpected checkpoint-diagnostic study type",
    )
    require(
        report["checkpoint_sha256"]
        == "66c3cf45b1476b40e3e65a84ea6106b51ca911a665d9bf145bb46314eec7302e",
        "checkpoint diagnostic references an unexpected checkpoint",
    )
    require(report["subset"]["seed"] == 20260825, "diagnostic subset seed drift")
    require(report["subset"]["per_ban"] == 100, "diagnostic subset size drift")
    conditions = report["conditions"]
    expected_conditions = {
        "full",
        "node_si_zero",
        "node_semantic_zero",
        "edge_operators_zero",
        "edge_role_direction_zero",
        "edge_flags_zero",
        "reverse_edges_removed",
    }
    require(set(conditions) == expected_conditions, "diagnostic condition set drift")
    for name, result in conditions.items():
        require(
            all(result["ladder"][f"L{i}"]["n"] == 100 for i in range(5)),
            f"unexpected ladder sample count in {name}",
        )
    require(
        conditions["full"]["aggregate"]["mae_log10_scaled"] < 0.01,
        "full-checkpoint diagnostic MAE drift",
    )
    require(
        conditions["edge_role_direction_zero"]["aggregate"]["mae_log10_scaled"]
        > 10 * conditions["full"]["aggregate"]["mae_log10_scaled"],
        "edge role/direction diagnostic no longer shows the archived degradation",
    )


def validate_flat_baseline() -> None:
    report = load_json("experiments/v1_1_flat_mlp_baseline.json")
    require(
        report["study_type"]
        == "multi-seed flat MLP baseline on the fixed ladder exam",
        "unexpected flat-baseline study type",
    )
    require(report["model_parameters"] == 1916637, "flat MLP parameter-count drift")
    require(report["data"]["training_rows_used"] == 617500, "training split drift")
    require(report["data"]["validation_rows_excluded"] == 32500, "validation split drift")
    require(report["data"]["validation_rows_evaluated"] == 10000, "validation subset drift")
    require(
        [row["seed"] for row in report["seeds"]] == [7, 17, 29],
        "baseline seeds drift",
    )
    require(
        all(row["test_evaluated_once_after_validation_selection"] for row in report["seeds"]),
        "baseline test-selection policy drift",
    )
    for ladder in (f"L{i}" for i in range(5)):
        values = [row["ladder"][ladder]["r2"] for row in report["seeds"]]
        summary = report["summary"]["ladder"][ladder]
        require(
            math.isclose(summary["r2_mean"], sum(values) / len(values), abs_tol=1e-12),
            f"baseline {ladder} mean is internally inconsistent",
        )
        oracle = report["reference_baselines"]["analytic_formula_oracle"]["ladder"][
            ladder
        ]["r2"]
        require(math.isclose(oracle, 1.0, abs_tol=1e-10), f"{ladder} oracle drift")


def validate_manuscript_source() -> None:
    source = (ROOT / "paper/physics_gnn_symbolic_discovery.tex").read_text(
        encoding="utf-8"
    )
    require("[MLP" not in source, "unresolved MLP placeholder in manuscript")
    require("10.5281/zenodo.21984785" in source, "stable concept DOI missing")
    require("a world can be represented as a graph" in source, "author graph perspective missing")
    require("low-income family in Bihar" in source, "author background statement missing")
    require("node features" in source.lower(), "node features are not discussed")
    require("edge features" in source.lower(), "edge features are not discussed")


def validate_hashes() -> None:
    for relative_path, expected in EXPECTED_HASHES.items():
        actual = sha256(ROOT / relative_path)
        require(actual == expected, f"SHA-256 mismatch for {relative_path}: {actual}")


def main() -> int:
    checks = [
        validate_graph,
        validate_history,
        validate_discoveries,
        validate_checkpoint_diagnostics,
        validate_flat_baseline,
        validate_manuscript_source,
        validate_hashes,
    ]
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
