"""
DERIVATION CHECKER v1 — symbolic proofs + prior-shift artifact test
====================================================================
Classifies every discovered connection A -> B into:

  PROVEN PURE       exact law B = C * A^k          (proof printed)
  PROVEN CONDITIONAL exact law B = C * A^k * f(extra vars)  (proof)
  BAYESIAN ARTIFACT  no derivation + exponent moves when data priors
                     shift  => model inference under sampling priors
  ADDITIVE-MEDIATED  A,B linked only through non-monomial equations
                     (add/trig/log) — outside this checker's algebra
  UNRESOLVED         none of the above fired cleanly

METHOD 1 — log-space Gaussian elimination (exact, Fractions):
  every multiplicative equation is linear in log-magnitudes:
      F = m a          ->  x_F - x_m - x_a = 0
      v_esc=sqrt(2GM/r)->  x_vesc - (x_G+x_M-x_r)/2 - log10(sqrt2) = 0
  Stack ~78 monomial equations, eliminate all variables except
  {A, B, physical constants}; a surviving row IS the law, with exact
  rational exponent and the list of equations used = a proof.

METHOD 2 — prior-shift invariance:
  regenerate small datasets with NARROW and SHIFTED sampling ranges
  (monkey-patched generator).  True laws: k_data identical.  Bayesian
  artifacts: k_data moves, because it was always a prior statistic.

RUN:   python derivation_checker_v1.py            (<1 min, CPU only)
       reads discoveries_v1.json if present; controls always checked.
       options: --pair Q_a Q_b   --extras 2
"""
import os, sys, json, math, argparse
from fractions import Fraction as Fr
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pair", nargs=2)
ap.add_argument("--extras", type=int, default=2)
ap.add_argument("--top", type=int, default=40)
args, _ = ap.parse_known_args()

L = lambda v: math.log10(v)
R2, R3, PI = L(2), L(3), L(math.pi)

CONSTS = {"Q_g","Q_G","Q_R_gas","Q_kB","Q_P_atm","Q_sigma_SB"}

# ── monomial system: eq_id -> (out, {in: exponent}, log10 coeff) ────
H = Fr(1,2)
EQM = {
 "EQ_vavg_st":("Q_v_avg",{"Q_s":1,"Q_t":-1},0),
 "EQ_x_t":("Q_x",{"Q_ux":1,"Q_t":1},0),
 "EQ_vx_c":("Q_vx",{"Q_ux":1},0),
 "EQ_hmax":("Q_H_max",{"Q_uy":2,"Q_g":-1},-R2),
 "EQ_tflight":("Q_T_flight",{"Q_uy":1,"Q_g":-1},R2),
 "EQ_vtan":("Q_v_tan",{"Q_omega":1,"Q_r":1},0),
 "EQ_ac":("Q_ac",{"Q_v_tan":2,"Q_r":-1},0),
 "EQ_om_avg":("Q_omega",{"Q_phi":1,"Q_t":-1},0),
 "EQ_T_om":("Q_T_period",{"Q_omega":-1},L(2*math.pi)),
 "EQ_f_T":("Q_f",{"Q_T_period":-1},0),
 "EQ_om_f":("Q_omega",{"Q_f":1},L(2*math.pi)),
 "EQ_atan":("Q_a_tan",{"Q_alpha":1,"Q_r":1},0),
 "EQ_F_ma":("Q_F",{"Q_m":1,"Q_a":1},0),
 "EQ_weight":("Q_W_wt",{"Q_m":1,"Q_g":1},0),
 "EQ_fric":("Q_f_fric",{"Q_mu":1,"Q_N_norm":1},0),
 "EQ_p_mv":("Q_p",{"Q_m":1,"Q_v":1},0),
 "EQ_F_dpt":("Q_F",{"Q_dp":1,"Q_t":-1},0),
 "EQ_J_Ft":("Q_J_imp",{"Q_F":1,"Q_t":1},0),
 "EQ_J_dp":("Q_dp",{"Q_J_imp":1},0),
 "EQ_KE":("Q_KE",{"Q_m":1,"Q_v":2},-R2),
 "EQ_PEg":("Q_PE_g",{"Q_m":1,"Q_g":1,"Q_h":1},0),
 "EQ_PEspr":("Q_PE_spr",{"Q_k_spr":1,"Q_x_spr":2},-R2),
 "EQ_Fspr":("Q_F_spr",{"Q_k_spr":1,"Q_x_spr":1},0),
 "EQ_WE_thm":("Q_dKE",{"Q_W_work":1},0),
 "EQ_P_Wt":("Q_P_pow",{"Q_W_work":1,"Q_t":-1},0),
 "EQ_P_Fv":("Q_P_pow",{"Q_F":1,"Q_v":1},0),
 "EQ_dEm_Wnc":("Q_dE_mech",{"Q_W_work":1},0),
 "EQ_tau_Ia":("Q_tau",{"Q_I_rot":1,"Q_alpha":1},0),
 "EQ_I_mr2":("Q_I_rot",{"Q_m":1,"Q_r":2},0),
 "EQ_L_Iw":("Q_L_ang",{"Q_I_rot":1,"Q_omega":1},0),
 "EQ_L_mvr":("Q_L_ang",{"Q_m":1,"Q_v":1,"Q_r":1},0),
 "EQ_tau_dLt":("Q_tau",{"Q_dL":1,"Q_t":-1},0),
 "EQ_KErot":("Q_KE_rot",{"Q_I_rot":1,"Q_omega":2},-R2),
 "EQ_roll_v":("Q_v",{"Q_omega":1,"Q_r":1},0),
 "EQ_L_after":("Q_L_ang",{"Q_I2":1,"Q_om2":1},0),
 "EQ_Fgrav":("Q_F_grav",{"Q_G":1,"Q_M_big":1,"Q_m":1,"Q_r_orb":-2},0),
 "EQ_gfield":("Q_g_field",{"Q_G":1,"Q_M_big":1,"Q_r_orb":-2},0),
 "EQ_PEorb":("Q_PE_orb",{"Q_G":1,"Q_M_big":1,"Q_m":1,"Q_r_orb":-1},0),
 "EQ_vorb":("Q_v_orb",{"Q_G":H,"Q_M_big":H,"Q_r_orb":-H},0),
 "EQ_Torb":("Q_T_orb",{"Q_r_orb":Fr(3,2),"Q_G":-H,"Q_M_big":-H},
            L(2*math.pi)),
 "EQ_vesc":("Q_v_esc",{"Q_G":H,"Q_M_big":H,"Q_r_orb":-H},R2/2),
 "EQ_F_mgf":("Q_F_grav",{"Q_m":1,"Q_g_field":1},0),
 "EQ_Eorb":("Q_E_orb",{"Q_G":1,"Q_M_big":1,"Q_m":1,"Q_r_orb":-1},-R2),
 "EQ_a_x":("Q_a_shm",{"Q_omega":2,"Q_x_shm":1},0),
 "EQ_om_km":("Q_omega",{"Q_k_spr":H,"Q_m":-H},0),
 "EQ_T_spr":("Q_T_period",{"Q_m":H,"Q_k_spr":-H},L(2*math.pi)),
 "EQ_T_pend":("Q_T_period",{"Q_L_pend":H,"Q_g":-H},L(2*math.pi)),
 "EQ_E_kA":("Q_E_shm",{"Q_k_spr":1,"Q_A_amp":2},-R2),
 "EQ_vmax":("Q_v_max",{"Q_A_amp":1,"Q_omega":1},0),
 "EQ_amax":("Q_a_max",{"Q_A_amp":1,"Q_omega":2},0),
 "EQ_P_FA":("Q_P_press",{"Q_F":1,"Q_A_area":-1},0),
 "EQ_rho_mV":("Q_rho",{"Q_m":1,"Q_V_vol":-1},0),
 "EQ_m_rhoV":("Q_m",{"Q_rho":1,"Q_V_vol":1},0),
 "EQ_buoy":("Q_F_buoy",{"Q_rho":1,"Q_V_vol":1,"Q_g":1},0),
 "EQ_Qflow":("Q_Q_flow",{"Q_A_area":1,"Q_v_fl":1},0),
 "EQ_torr":("Q_v_fl",{"Q_g":H,"Q_h":H},R2/2),
 "EQ_gas":("Q_P_press",{"Q_n_mol":1,"Q_R_gas":1,"Q_T_temp":1,
           "Q_V_vol":-1},0),
 "EQ_W_PdV":("Q_W_th",{"Q_P_press":1,"Q_dV":1},0),
 "EQ_Q_mcdT":("Q_Q_heat",{"Q_m":1,"Q_c_spec":1,"Q_dT":1},0),
 "EQ_Q_mL":("Q_Q_heat",{"Q_m":1,"Q_L_lat":1},0),
 "EQ_eta_WQ":("Q_eta",{"Q_W_th":1,"Q_Q_heat":-1},0),
 "EQ_dS":("Q_dS",{"Q_Q_heat":1,"Q_T_temp":-1},0),
 "EQ_U_nRT":("Q_U_int",{"Q_n_mol":1,"Q_R_gas":1,"Q_T_temp":1},
             L(1.5)),
 "EQ_dU_nRdT":("Q_dU",{"Q_n_mol":1,"Q_R_gas":1,"Q_dT":1},L(1.5)),
 "EQ_vrms":("Q_v_rms",{"Q_R_gas":H,"Q_T_temp":H,"Q_M_molar":-H},R3/2),
 "EQ_KEavg":("Q_KE_avg",{"Q_kB":1,"Q_T_temp":1},L(1.5)),
 "EQ_v_flam":("Q_v_wave",{"Q_f":1,"Q_lambda":1},0),
 "EQ_v_str":("Q_v_wave",{"Q_T_tens":H,"Q_mu_lin":-H},0),
 "EQ_kwave":("Q_k_wave",{"Q_lambda":-1},L(2*math.pi)),
 "EQ_I_PA":("Q_I_int",{"Q_P_pow":1,"Q_A_area":-1},0),
 "EQ_stand":("Q_lambda",{"Q_L_str":1,"Q_n_harm":-1},R2),
 "EQ_f_fund":("Q_f",{"Q_v_wave":1,"Q_L_str":-1},-R2),
 "EQ_v_sound":("Q_v_wave",{"Q_gamma":H,"Q_R_gas":H,"Q_T_temp":H,
               "Q_M_molar":-H},0),
 "EQ_v_omk":("Q_v_wave",{"Q_omega":1,"Q_k_wave":-1},0),
 "EQ_stress":("Q_sigma_s",{"Q_F":1,"Q_A_area":-1},0),
 "EQ_strain":("Q_eps_str",{"Q_dL_el":1,"Q_L0":-1},0),
 "EQ_young":("Q_Y_mod",{"Q_sigma_s":1,"Q_eps_str":-1},0),
 "EQ_U_el":("Q_U_el",{"Q_F":1,"Q_dL_el":1},-R2),
 "EQ_stokes":("Q_F_visc",{"Q_eta_visc":1,"Q_r":1,"Q_v":1},
              L(6*math.pi)),
 "EQ_dP_surf":("Q_P_press",{"Q_S_surf":1,"Q_r":-1},R2),
 "EQ_thermx":("Q_dL_el",{"Q_alpha_x":1,"Q_L0":1,"Q_dT":1},0),
 "EQ_conduct":("Q_P_cond",{"Q_k_therm":1,"Q_A_area":1,"Q_dT":1,
               "Q_L0":-1},0),
 "EQ_stefan":("Q_P_rad",{"Q_sigma_SB":1,"Q_A_area":1,"Q_T_temp":4},0),
}
# non-monomial equations touching each quantity (for mediation notes)
NONMONO = {
 "additive/trig": ["EQ_v_uat","EQ_s_ut2","EQ_v2_2as","EQ_s_uvt",
  "EQ_s_vt2","EQ_vavg_uv","EQ_ux","EQ_uy","EQ_y_t","EQ_vy_t",
  "EQ_range","EQ_speed_xy","EQ_phi_t2","EQ_om_lin","EQ_om_sq",
  "EQ_atot","EQ_N_incl","EQ_F_incl","EQ_W_Fd","EQ_Emech","EQ_E_split",
  "EQ_v_Ax","EQ_xshm","EQ_vshm_t","EQ_ashm_t","EQ_ptot_i","EQ_ptot_f",
  "EQ_vcm","EQ_v1f_el","EQ_v2f_el","EQ_v_pi","EQ_e_rest","EQ_KElost",
  "EQ_xcm","EQ_tau_rF","EQ_KEroll","EQ_parax","EQ_P_hydro","EQ_bern",
  "EQ_1law","EQ_carnot","EQ_W_iso","EQ_ywave","EQ_beat","EQ_doppler",
  "EQ_I_A2","EQ_vterm","EQ_capil"]}

def build_rows():
    rows = []
    for eid,(out, ins, c) in EQM.items():
        r = {out: Fr(1)}
        for q,e in ins.items():
            r[q] = r.get(q,Fr(0)) - Fr(e)
        rows.append((dict(r), float(-c), {eid: Fr(1)}))
    return rows

def eliminate(A, B, extras):
    keep = {A,B} | CONSTS | set(extras)
    rows = build_rows()
    allv = sorted({v for r,_,_ in rows for v in r})
    for v in allv:
        if v in keep: continue
        piv = next((i for i,(r,_,_) in enumerate(rows)
                    if r.get(v,Fr(0)) != 0), None)
        if piv is None: continue
        pr, pc, pp = rows.pop(piv)
        pv = pr[v]
        out = []
        for r,c,p in rows:
            f = r.get(v,Fr(0))
            if f == 0: out.append((r,c,p)); continue
            s = f/pv
            nr = {k: r.get(k,Fr(0)) - s*pr.get(k,Fr(0))
                  for k in set(r)|set(pr)}
            nr = {k:x for k,x in nr.items() if x != 0}
            np_ = {k: p.get(k,Fr(0)) - s*pp.get(k,Fr(0))
                   for k in set(p)|set(pp)}
            out.append((nr, c - float(s)*pc,
                        {k:x for k,x in np_.items() if x!=0}))
        rows = out
    for r,c,p in rows:
        supp = {k for k in r if k not in CONSTS}
        if supp == {A,B} or supp == {A,B}|set(extras):
            if r.get(A) and r.get(B):
                k = -r[A]/r[B]
                # solve row for B: B = -(others)/r[B]
                coefB = r[B]
                pref = -c/float(coefB)
                ex = {q: -r.get(q,Fr(0))/coefB
                      for q in (set(extras)|CONSTS) if r.get(q)}
                return dict(k=k, log10_pref=pref, extras=ex, proof=p)
    return None

def nice(x):
    f = float(x)
    for val,s in [(1,""),(2,"2"),(.5,"1/2"),(3,"3"),(1.5,"3/2"),
                  (math.pi,"pi"),(2*math.pi,"2pi"),(2**.5,"sqrt2"),
                  (3**.5,"sqrt3"),(.25,"1/4"),(4,"4"),(1/3,"1/3"),
                  (6*math.pi,"6pi"),(1/(2*math.pi),"1/2pi"),(8,"8"),
                  (1/8,"1/8"),(2/9,"2/9"),((2*math.pi)**(1/3.),"(2pi)^1/3"),((2*math.pi)**(2/3.),"(2pi)^2/3"),(4*math.pi**2,"4pi^2"),(1/(4*math.pi**2),"1/4pi^2")]:
        if abs(f - L(val) if val!=1 else abs(f)) < 1e-6:
            return s or "1"
    return f"10^{f:.3f}"

def fmt(res, A, B):
    k = res["k"]
    terms = [f"{A}^{k}"]
    for q,e in res["extras"].items():
        if e == 0: continue
        terms.append(f"{q}^{e}")
    pf = nice(res["log10_pref"])
    proof = " ".join(f"{'+' if f>0 else '-'}{abs(f)}x{e}"
                     for e,f in sorted(res["proof"].items()))
    return (f"{B} = {pf} * " + " * ".join(terms), proof)

# ════════════════════════════════════════════════════════════════════
# METHOD 2 — prior-shift data regeneration (monkey-patched generator)
# ════════════════════════════════════════════════════════════════════
import scenario_generator_v1 as SG

import hashlib
def _jit(salt, a, b):
    """deterministic 0..1 per variable-range: breaks variance ratios"""
    return hashlib.md5(f"{salt}|{a:.6g}|{b:.6g}".encode())                   .digest()[0] / 255.0

def gen_kdata(mode, pairs, n_per=1200, seed=11):
    oLU, oU = SG.LU, SG.U
    salt = mode
    def LU2(a, b):
        j = _jit(salt, a, b)
        span = math.log(b/a)
        w    = 0.25 + 0.35*j          # each var keeps 25-60% of width
        lo   = a*math.exp(span*(0.05 + 0.55*j))
        return oLU(lo, lo*math.exp(span*w))
    def U2(a, b):
        j = _jit(salt, a, b); w = b-a
        lo = a + w*(0.05 + 0.5*j)
        return oU(lo, min(b, lo + w*(0.25+0.3*j)))
    SG.LU, SG.U = LU2, U2
    SG.rng = np.random.default_rng(seed)
    NQ = len(SG.QTY)
    mags = np.zeros((n_per*len(SG.SCENARIOS), NQ), np.float32)
    ex   = np.zeros_like(mags, bool)
    row = 0
    for name, fn in SG.SCENARIOS:
        for _ in range(n_per):
            w = fn()
            for q,val in w.items():
                mags[row, SG.QI[q]] = SG.mag(val); ex[row, SG.QI[q]]=True
            row += 1
    SG.LU, SG.U = oLU, oU
    out = {}
    for a,b in pairs:
        both = ex[:,a] & ex[:,b]
        if both.sum() < 300 or mags[both,a].std() < 1e-4:
            out[(a,b)] = None; continue
        x,y = mags[both,a], mags[both,b]
        out[(a,b)] = float(np.cov(x,y)[0,1]/x.var())
    return out

# ════════════════════════════════════════════════════════════════════
# VERDICT ENGINE
# ════════════════════════════════════════════════════════════════════
QTY = SG.QTY; QC = {q:i for i,q in enumerate(QTY)}

def check_pair(A, B, k_probe=None):
    r = eliminate(A, B, [])
    tag, law, proof, extras = None, None, None, []
    if r: tag = "PROVEN PURE"
    else:
        allv = {v for eq,(o,i,c) in EQM.items()
                for v in [o]+list(i)} - CONSTS - {A,B}
        prio = ["Q_m","Q_r","Q_t","Q_A_amp","Q_M_big","Q_M_molar",
                "Q_omega","Q_k_spr","Q_r_orb","Q_T_temp","Q_v",
                "Q_L_str","Q_rho","Q_n_mol","Q_A_area"]
        cand = [q for q in prio if q in allv] +                sorted(allv - set(prio))
        for e1 in cand:
            r = eliminate(A, B, [e1])
            if r: tag, extras = "PROVEN CONDITIONAL", [e1]; break
        if not r and args.extras >= 2:
            for i1 in range(len(cand)):
                for e2 in cand[i1+1:]:
                    r = eliminate(A, B, [cand[i1], e2])
                    if r:
                        tag, extras = "PROVEN CONDITIONAL", \
                                      [cand[i1], e2]
                        break
                if r: break
    if r:
        law, proof = fmt(r, A, B)
        return dict(tag=tag, k_exact=str(r["k"]),
                    k_float=float(r["k"]), extras=extras,
                    law=law, proof=proof)
    inA = any(A in (o,*i) for o,i,_ in EQM.values())
    inB = any(B in (o,*i) for o,i,_ in EQM.values())
    note = []
    if not inA: note.append(f"{A} appears only in additive/trig eqs")
    if not inB: note.append(f"{B} appears only in additive/trig eqs")
    return dict(tag="NOT-DERIVABLE(monomial)",
                note="; ".join(note) or "no monomial chain")

def main():
    print("="*68)
    print("SELF-TEST — controls (must recover exact laws)")
    print("="*68)
    CTRL = [("Q_v_orb","Q_v_esc"),("Q_T_flight","Q_H_max"),
            ("Q_p","Q_KE"),("Q_v_max","Q_a_max"),
            ("Q_T_orb","Q_v_orb"),("Q_KE_avg","Q_v_rms"),
            ("Q_e_rest","Q_S_surf"),("Q_n_harm","Q_PE_g")]
    for A,B in CTRL:
        c = check_pair(A,B)
        line = f"{A} -> {B:14s} {c['tag']}"
        if "k_exact" in c:
            line += f"  k={c['k_exact']}"
            if c["extras"]: line += f"  given {c['extras']}"
            line += f"\n    LAW: {c['law']}"
        else:
            line += f"  ({c.get('note','')})"
        print(line)

    if args.pair:
        A,B = args.pair
        c = check_pair(A,B)
        print(json.dumps(c, indent=1)); return

    if not os.path.exists("discoveries_v1.json"):
        print("\n(no discoveries_v1.json found — controls only)"); return
    disc = json.load(open("discoveries_v1.json"))["candidates"]
    disc = disc[:args.top]
    pairs_idx = [(QC[d["A"]], QC[d["B"]]) for d in disc]
    print(f"\nprior-shift regeneration (narrow + shifted) ...")
    k_nar = gen_kdata("narrow", pairs_idx)
    k_shf = gen_kdata("shift",  pairs_idx, seed=12)

    print("\n" + "="*68)
    print("VERDICTS ON DISCOVERED CANDIDATES")
    print("="*68)
    out = []
    for d,(ai,bi) in zip(disc, pairs_idx):
        c = check_pair(d["A"], d["B"])
        kd0, kn, ks = d.get("k_data"), k_nar[(ai,bi)], k_shf[(ai,bi)]
        stable = (kd0 is not None and kn is not None and ks is not None
                  and abs(kd0-kn) < 0.12 and abs(kd0-ks) < 0.12)
        if "PROVEN" in c["tag"]:
            verdict = c["tag"]
        elif kd0 is not None and not stable:
            verdict = "BAYESIAN ARTIFACT (k moves with priors)"
        elif c["tag"].startswith("NOT") and "additive" in \
             c.get("note",""):
            verdict = "ADDITIVE-MEDIATED"
        elif stable:
            verdict = "UNRESOLVED-INVARIANT (inspect!)"
        else:
            verdict = "UNRESOLVED"
        rec = dict(A=d["A"], B=d["B"], probe_k=d["k"],
                   verdict=verdict, **c,
                   k_data=kd0, k_narrow=kn, k_shift=ks)
        out.append(rec)
        line = (f"{d['A']+' -> '+d['B']:34s} probe_k={d['k']:+.2f}  "
                f"{verdict}")
        if "k_exact" in c:
            line += f"\n    exact k={c['k_exact']}"
            if c.get("extras"): line += f" given {c['extras']}"
            line += f" | attenuation {d['k']/max(c['k_float'],1e-9):+.2f}"
            line += f"\n    LAW: {c['law']}"
        if kd0 is not None:
            line += (f"\n    k_data orig={kd0:+.2f} "
                     f"narrow={kn if kn is None else round(kn,2)} "
                     f"shift={ks if ks is None else round(ks,2)}")
        print(line)
    json.dump(out, open("checker_v1.json","w"), indent=1)
    n = {}
    for r in out: n[r["verdict"].split()[0]] = \
        n.get(r["verdict"].split()[0],0)+1
    print("\nSUMMARY:", ", ".join(f"{k}={v}" for k,v in n.items()))
    print("saved: checker_v1.json")

if __name__ == "__main__":
    main()
