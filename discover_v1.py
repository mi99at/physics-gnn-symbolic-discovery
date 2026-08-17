"""
DISCOVERY MINER v1 — counterfactual probing of the trained physics GNN
=======================================================================
GOAL  Find quantity pairs (A -> B) with NO encoded connection (graph
      distance >= 3) where the trained model shows a LAWFUL response:
      sweep A's value, read model's prediction of B.  In log-magnitude
      space a straight line of slope k means  B ∝ A^k  — the slope IS
      the discovered logic (k=+2 square, -2 inverse-square, +0.5 sqrt).

HONESTY PROTOCOL (runs first, automatically)
  Positive controls: true physics identities that are NOT edges in our
  graph (e.g. KE = p^2/2m, v_esc = sqrt(2) v_orb, H = g T_f^2 / 8).
  The miner must recover their known exponents.
  Negative controls: unrelated pairs — must show ~zero effect.
  If controls fail, trust nothing below them.

PIPELINE
  1. validation controls (report table)
  2. fast screen: ALL eligible pairs, A-alone probing (batched)
  3. refinement: top pairs re-probed in real scenario contexts
  4. data cross-check where A,B co-occur in training worlds
  5. report: discoveries_report.txt + discoveries_v1.json

RUN (Colab, after training has a checkpoint):
  %cd /content/drive/MyDrive/physics_gnn_v9
  !python discover_v1.py                      # ~10-20 min on T4
Options: --limit N (screen only N pairs)  --pair Q_a Q_b (probe one)
"""
import sys, os, json, math, argparse
import numpy as np
import torch

import gnn_trainer_v9 as T          # reuse model, graph, data, tensors

argp = argparse.ArgumentParser()
argp.add_argument("--limit", type=int, default=0)
argp.add_argument("--pair", nargs=2, default=None)
argp.add_argument("--ckpt", default=None)
args, _ = argp.parse_known_args()

DEV, NODES, NQ = T.DEV, T.NODES, T.NQ
QTY, QCOL = T.QTY_IDS, {q:i for i,q in enumerate(T.QTY_IDS)}
QNODE, CONSTt = T.QNODE, T.CONSTt
MAGS, EXIST = T.MAGS, T.EXIST
EPS = 1e-12
def magf(x): return math.log10(abs(x)+EPS)/10.0

meta = json.load(open("scenario_meta_v1.json"))
CONST_MAG = torch.zeros(NQ, device=DEV)
for q, v in meta["constants"].items():
    CONST_MAG[QCOL[q]] = magf(v)

# ── load trained model ──────────────────────────────────────────────
model = T.PhysicsGNN().to(DEV)
ck = args.ckpt or os.path.join(T.CKDIR, "v9_best.pt")
if not os.path.exists(ck): ck = os.path.join(T.CKDIR, "v9_latest.pt")
if os.path.exists(ck):
    st = torch.load(ck, map_location=DEV, weights_only=False)
    model.load_state_dict(st["model"])
    print(f"loaded {ck} (epoch {st['epoch']}, best={st['best']:.4f})")
else:
    print("!! NO CHECKPOINT — random weights, results are noise (smoke only)")
model.eval()

# ── eligible pairs: graph distance >= 3, no constants ───────────────
from collections import defaultdict, deque
adj = defaultdict(set)
for s, d in zip(*T.EI_np):
    adj[int(s)].add(int(d)); adj[int(d)].add(int(s))
QN = T.QNODE_np
def bfs(src, cap=3):
    dist = {src:0}; dq = deque([src])
    while dq:
        n = dq.popleft()
        if dist[n] >= cap: continue
        for nb in adj[n]:
            if nb not in dist:
                dist[nb] = dist[n]+1; dq.append(nb)
    return dist
qdist = {}
for i, na in enumerate(QN):
    d = bfs(int(na), cap=2)                    # only need to know <=2
    for j, nb in enumerate(QN):
        if i != j: qdist[(i,j)] = d.get(int(nb), 9)
is_const = CONSTt.cpu().numpy()
mean_mag = np.array([MAGS[EXIST[:,c], c].mean() if EXIST[:,c].any()
                     else 0.0 for c in range(NQ)], np.float32)

PAIRS = [(a,b) for a in range(NQ) for b in range(NQ)
         if a!=b and not is_const[a] and not is_const[b]
         and qdist[(a,b)] >= 3]
print(f"eligible distant pairs: {len(PAIRS):,}")

# ── core probe: batched A-alone sweeps ──────────────────────────────
SW = torch.linspace(-0.25, 0.35, 9, device=DEV)     # sweep offsets
S  = len(SW)

@torch.no_grad()
def probe_alone(pairs):
    """pairs: list[(a,b)] -> slope k, linearity R2, effect, per pair"""
    out = []
    per = max(1, 256 // S)                          # pairs per forward
    for i0 in range(0, len(pairs), per):
        chunk = pairs[i0:i0+per]; B = len(chunk)*S
        mag_n = torch.zeros(B, NODES, device=DEV)
        sgn_n = torch.zeros(B, NODES, device=DEV)
        kn_n  = torch.zeros(B, NODES, device=DEV)
        qn    = QNODE.unsqueeze(0).expand(B, -1)
        # constants revealed everywhere
        cm = CONST_MAG.unsqueeze(0).expand(B, -1)
        mag_q = cm.clone(); kn_q = CONSTt.float().unsqueeze(0) \
                                     .expand(B, -1).clone()
        sg_q  = torch.ones(B, NQ, device=DEV)
        for pi, (a, b) in enumerate(chunk):
            rows = slice(pi*S, (pi+1)*S)
            mag_q[rows, a] = mean_mag[a] + SW
            kn_q[rows, a]  = 1.0
        mag_n.scatter_(1, qn, mag_q*kn_q)
        sgn_n.scatter_(1, qn, sg_q*kn_q)
        kn_n.scatter_(1, qn, kn_q)
        ei = T.batch_graph(B)
        with torch.autocast(DEV, enabled=T.AMP):
            pm, _ = model(mag_n, sgn_n, kn_n, B, ei)
        pm_q = pm.view(B, NODES).gather(1, qn).float()
        x = SW.cpu().numpy()
        for pi, (a, b) in enumerate(chunk):
            y = pm_q[pi*S:(pi+1)*S, b].cpu().numpy()
            k, c = np.polyfit(x, y, 1)
            res = y - (k*x + c)
            ss  = ((y - y.mean())**2).sum()
            r2  = 1 - (res**2).sum()/max(ss, 1e-12)
            out.append(dict(a=int(a), b=int(b), k=float(k),
                            lin=float(r2), eff=float(y.max()-y.min())))
    return out

@torch.no_grad()
def probe_context(a, b, n_worlds=16):
    """slope of B-pred vs A within real partial worlds of B."""
    rows = np.where(EXIST[:, b])[0]
    if len(rows) == 0: return None
    rows = np.random.default_rng(0).choice(rows,
              min(n_worlds, len(rows)), replace=False)
    ks = []
    for r in rows:
        ex = EXIST[r]; B = S
        rev = ex & (np.random.default_rng(r).random(NQ) < .5)
        rev |= is_const & ex; rev[b] = False; rev[a] = False
        mag_q = torch.tensor(MAGS[r], device=DEV) \
                    .unsqueeze(0).repeat(B,1)
        kn_q  = torch.tensor(rev, device=DEV, dtype=torch.float32) \
                    .unsqueeze(0).repeat(B,1)
        sg_q  = torch.tensor((T.SIGNS[r]!=0)*2-1., device=DEV,
                    dtype=torch.float32).unsqueeze(0).repeat(B,1)
        mag_q[:, a] = mean_mag[a] + SW; kn_q[:, a] = 1.
        sg_q[:, a]  = 1.
        mag_n = torch.zeros(B, NODES, device=DEV)
        sgn_n = torch.zeros(B, NODES, device=DEV)
        kn_n  = torch.zeros(B, NODES, device=DEV)
        qn = QNODE.unsqueeze(0).expand(B, -1)
        mag_n.scatter_(1, qn, mag_q*kn_q)
        sgn_n.scatter_(1, qn, sg_q*kn_q)
        kn_n.scatter_(1, qn, kn_q)
        with torch.autocast(DEV, enabled=T.AMP):
            pm, _ = model(mag_n, sgn_n, kn_n, B, T.batch_graph(B))
        y = pm.view(B, NODES).gather(1, qn)[:, b].float().cpu().numpy()
        ks.append(np.polyfit(SW.cpu().numpy(), y, 1)[0])
    ks = np.array(ks)
    return dict(k_ctx=float(np.median(ks)),
                iqr=float(np.percentile(ks,75)-np.percentile(ks,25)))

def data_check(a, b):
    both = EXIST[:, a] & EXIST[:, b]
    if both.sum() < 500: return None
    x, y = MAGS[both, a], MAGS[both, b]
    if x.std() < 1e-4: return None
    k = float(np.cov(x, y)[0,1]/x.var())
    r = float(np.corrcoef(x, y)[0,1]**2)
    return dict(k_data=k, r2_data=r)

def logic_label(k):
    for kk, lab in [(2,"SQUARE"),(1,"LINEAR"),(0.5,"SQRT"),(3,"CUBIC"),
                    (-1,"INVERSE"),(-2,"INV-SQUARE"),(-0.5,"INV-SQRT"),
                    (-1/3,"INV-CBRT"),(1/3,"CBRT"),(-3,"INV-CUBIC")]:
        if abs(k-kk) < 0.12: return lab
    return f"POWER k={k:+.2f}"

# ════════════════════════════════════════════════════════════════════
# VALIDATION CONTROLS — known truths NOT encoded as edges
# ════════════════════════════════════════════════════════════════════
POS = [  # (A, B, true k, note)
 ("Q_v_orb","Q_v_esc",  1.0, "v_esc = sqrt2 * v_orb"),
 ("Q_T_flight","Q_H_max",2.0,"H = g*Tf^2/8"),
 ("Q_p","Q_KE",          2.0,"KE = p^2/2m (m unknown -> partial)"),
 ("Q_v_max","Q_a_max",   2.0,"a_max = v_max^2/A (A unknown)"),
 ("Q_T_orb","Q_v_orb",  -1/3,"v_orb ∝ T^-1/3"),
 ("Q_KE_avg","Q_v_rms",  0.5,"v_rms ∝ sqrt(KE_avg)"),
]
NEG = [("Q_e_rest","Q_S_surf"),("Q_n_harm","Q_PE_g"),
       ("Q_theta","Q_M_molar"),("Q_f_beat","Q_Y_mod"),
       ("Q_x1","Q_L_lat"),("Q_mu","Q_T_orb")]

def run_controls():
    print("\n" + "="*68)
    print("VALIDATION — must pass before trusting discoveries")
    print("="*68)
    res = probe_alone([(QCOL[a], QCOL[b]) for a,b,_,_ in POS])
    okp = 0
    print(f"{'A -> B':34s} {'true k':>7s} {'found k':>8s} "
          f"{'lin':>5s} {'eff':>6s}")
    for (a,b,kt,note), r in zip(POS, res):
        hit = abs(r["k"]-kt) < 0.25 and r["eff"] > 0.05
        okp += hit
        print(f"{a+' -> '+b:34s} {kt:+7.2f} {r['k']:+8.3f} "
              f"{r['lin']:5.2f} {r['eff']:6.3f} "
              f"{'PASS' if hit else 'weak'}  ({note})")
    resn = probe_alone([(QCOL[a], QCOL[b]) for a,b in NEG])
    okn = sum(1 for r in resn if r["eff"] < 0.05)
    for (a,b), r in zip(NEG, resn):
        print(f"{a+' -> '+b:34s} {'0':>7s} {r['k']:+8.3f} "
              f"{r['lin']:5.2f} {r['eff']:6.3f} "
              f"{'PASS' if r['eff']<0.05 else 'LEAK'}")
    print(f"\ncontrols: {okp}/{len(POS)} positives recovered, "
          f"{okn}/{len(NEG)} negatives clean")
    return okp, okn

# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════
def main():
    if args.pair:
        a, b = QCOL[args.pair[0]], QCOL[args.pair[1]]
        r = probe_alone([(a,b)])[0]
        c = probe_context(a, b); d = data_check(a, b)
        print(json.dumps(dict(r, logic=logic_label(r["k"]),
              ctx=c, data=d), indent=1))
        return

    okp, okn = run_controls()

    pairs = PAIRS[:args.limit] if args.limit else PAIRS
    print(f"\nSCREENING {len(pairs):,} distant pairs "
          f"(A alone -> B response) ...")
    res = probe_alone(pairs)
    for r in res:
        r["score"] = r["eff"] * max(r["lin"], 0)
    res.sort(key=lambda r: -r["score"])

    topN = res[:40]
    print(f"refining top {len(topN)} in real scenario contexts ...")
    report = []
    for r in topN:
        a, b = r["a"], r["b"]
        c = probe_context(a, b) or {}
        d = data_check(a, b) or {}
        report.append(dict(
            A=QTY[a], B=QTY[b], k=round(r["k"],3),
            logic=logic_label(r["k"]), linearity=round(r["lin"],3),
            effect=round(r["eff"],3),
            k_context=round(c.get("k_ctx", float("nan")),3)
                      if c else None,
            ctx_iqr=round(c.get("iqr", float("nan")),3) if c else None,
            k_data=round(d.get("k_data", float("nan")),3) if d else None,
            r2_data=round(d.get("r2_data", float("nan")),3) if d else None,
            cross_domain=d is None or not d))
    json.dump(dict(controls=dict(pos=okp, neg=okn),
                   candidates=report,
                   all_scores=[dict(A=QTY[r["a"]],B=QTY[r["b"]],
                       k=round(r["k"],3),eff=round(r["eff"],3),
                       lin=round(r["lin"],3)) for r in res[:400]]),
              open("discoveries_v1.json","w"), indent=1)

    lines = ["="*68, "TOP DISCOVERED CONNECTIONS (no encoded edge, "
             "graph distance >= 3)", "="*68,
             f"{'A -> B':36s} {'logic':>12s} {'k':>7s} {'lin':>5s} "
             f"{'eff':>6s} {'k_ctx':>7s} {'k_data':>7s}"]
    for r in report:
        lines.append(f"{r['A']+' -> '+r['B']:36s} {r['logic']:>12s} "
            f"{r['k']:+7.2f} {r['linearity']:5.2f} {r['effect']:6.3f} "
            f"{str(r['k_context']):>7s} {str(r['k_data']):>7s}")
    txt = "\n".join(lines)
    open("discoveries_report.txt","w").write(txt)
    print("\n"+txt)
    print("\nsaved: discoveries_report.txt, discoveries_v1.json")
    print("READ ME: trust a row only if (1) controls passed above, "
          "(2) lin>0.9, (3) k_ctx agrees with k, (4) if k_data exists "
          "it agrees. cross-domain rows (no k_data) are the most "
          "interesting — verify by hand with the actual formulas.")

if __name__ == "__main__":
    main()
