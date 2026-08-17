"""
PHYSICS COROLLARY CATALOG v1
============================
Audits the 400 retained discovery candidates, deduplicates certified
formulas, and writes a human-readable Markdown catalog plus JSON results.

Run `discover_v1.py` first. The final discovery JSON must contain both the
top candidates and the 400-deep score list.
"""
import json
import itertools
import time

import numpy as np

import derivation_checker_v1 as DC

EQM, CONSTS = DC.EQM, DC.CONSTS
SG = DC.SG
QI, QTY = SG.QI, SG.QTY

# Variables sharing monomial equations form a compact, physics-aware pool of
# conditioning variables. This avoids the alphabetical-search failure found in
# the first catalog pass.
CO = {}
for _eid, (out, ins, _const) in EQM.items():
    vars_ = [out] + list(ins)
    for var in vars_:
        CO.setdefault(var, set()).update(other for other in vars_ if other != var)

PRIORITY = [
    "Q_m", "Q_a", "Q_t", "Q_v", "Q_r", "Q_omega", "Q_alpha", "Q_F",
    "Q_A_amp", "Q_M_big", "Q_M_molar", "Q_k_spr", "Q_r_orb",
    "Q_T_temp", "Q_dT", "Q_P_press", "Q_v_fl", "Q_L_str", "Q_rho",
    "Q_n_mol", "Q_A_area", "Q_I_rot", "Q_h", "Q_x_shm", "Q_dp",
    "Q_tau", "Q_dL", "Q_lambda", "Q_f",
]


def nearby(a, b, hops=2):
    seen, frontier = {a, b}, {a, b}
    for _ in range(hops):
        frontier = {node for src in frontier for node in CO.get(src, ())} - seen
        seen |= frontier
    return (seen - CONSTS - {a, b}) & set(CO)


def ordered_candidates(a, b):
    one_hop = (CO.get(a, set()) | CO.get(b, set())) - CONSTS - {a, b}
    pool = nearby(a, b) | one_hop
    return sorted(
        pool,
        key=lambda q: (q not in one_hop, q not in PRIORITY, -len(CO.get(q, ())), q),
    ), one_hop


def fast_check(a, b, max_extras=3):
    if a not in CO or b not in CO:
        missing = [q for q in (a, b) if q not in CO]
        return {"tag": "ADDITIVE-MEDIATED", "note": f"{','.join(missing)} only in additive/trig equations"}

    result = DC.eliminate(a, b, [])
    if result:
        law, _ = DC.fmt(result, a, b)
        return {"tag": "PROVEN PURE", "k_exact": str(result["k"]), "k_float": float(result["k"]), "extras": [], "law": law, "n_proof": len(result["proof"])}

    candidates, one_hop = ordered_candidates(a, b)
    for extra in candidates[:48]:
        result = DC.eliminate(a, b, [extra])
        if result:
            law, _ = DC.fmt(result, a, b)
            return {"tag": "PROVEN CONDITIONAL", "k_exact": str(result["k"]), "k_float": float(result["k"]), "extras": [extra], "law": law, "n_proof": len(result["proof"])}

    if max_extras >= 2:
        for first, second in itertools.combinations(candidates[:32], 2):
            result = DC.eliminate(a, b, [first, second])
            if result:
                law, _ = DC.fmt(result, a, b)
                return {"tag": "PROVEN CONDITIONAL", "k_exact": str(result["k"]), "k_float": float(result["k"]), "extras": [first, second], "law": law, "n_proof": len(result["proof"])}

    if max_extras >= 3:
        pool = [q for q in candidates if q in one_hop or q in PRIORITY][:14]
        for first, second, third in itertools.combinations(pool, 3):
            result = DC.eliminate(a, b, [first, second, third])
            if result:
                law, _ = DC.fmt(result, a, b)
                return {"tag": "PROVEN CONDITIONAL", "k_exact": str(result["k"]), "k_float": float(result["k"]), "extras": [first, second, third], "law": law, "n_proof": len(result["proof"])}

    return {"tag": "NOT-DERIVABLE(monomial)"}


def main():
    started = time.time()
    discoveries = json.load(open("discoveries_v1.json"))
    seen, candidates = set(), []
    for row in discoveries.get("candidates", []) + discoveries.get("all_scores", []):
        key = (row["A"], row["B"])
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"A": row["A"], "B": row["B"], "probe_k": row["k"], "lin": row.get("linearity", row.get("lin")), "eff": row.get("effect", row.get("eff"))})

    print(f"unique candidate pairs: {len(candidates)}")
    data = np.load("scenario_data_v1.npz")
    mags, exists = data["mags"], data["exist"]

    def data_slope(a, b):
        both = exists[:, a] & exists[:, b]
        if both.sum() < 500 or mags[both, a].std() < 1e-4:
            return None
        x, y = mags[both, a], mags[both, b]
        return float(np.cov(x, y)[0, 1] / x.var())

    indexes = [(QI[row["A"]], QI[row["B"]]) for row in candidates]
    narrow = DC.gen_kdata("narrow", indexes)
    shifted = DC.gen_kdata("shift", indexes, seed=12)

    results = []
    for index, (candidate, (a, b)) in enumerate(zip(candidates, indexes), 1):
        check = fast_check(candidate["A"], candidate["B"])
        original = data_slope(a, b)
        k_narrow, k_shift = narrow[(a, b)], shifted[(a, b)]
        stable = (original is not None and k_narrow is not None and k_shift is not None and abs(original - k_narrow) < 0.12 and abs(original - k_shift) < 0.12)
        if "PROVEN" in check["tag"]:
            verdict = check["tag"]
            attenuation = candidate["probe_k"] / check["k_float"] if abs(check["k_float"]) > 1e-9 else None
        elif check["tag"] == "ADDITIVE-MEDIATED":
            verdict, attenuation = "ADDITIVE-MEDIATED", None
        elif original is not None and not stable:
            verdict, attenuation = "BAYESIAN ARTIFACT", None
        elif stable:
            verdict, attenuation = "UNRESOLVED-INVARIANT", None
        else:
            verdict, attenuation = "UNRESOLVED", None
        results.append({**candidate, **check, "verdict": verdict, "attenuation": None if attenuation is None else round(attenuation, 3), "k_data": original, "k_narrow": k_narrow, "k_shift": k_shift})
        if index % 50 == 0:
            print(f"  {index}/{len(candidates)}  ({time.time() - started:.0f}s)")

    counts = {}
    for row in results:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    by_law = {}
    for row in results:
        if row.get("law"):
            by_law.setdefault(row["law"], []).append(row)

    json.dump(results, open("catalog_v1.json", "w"), indent=1)
    print("results:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"unique proven formulas: {len(by_law)}")


if __name__ == "__main__":
    main()
