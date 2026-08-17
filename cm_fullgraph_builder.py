"""
CLASSICAL MECHANICS — FULL GRAPH v1.0 (multi-path optimized)
=============================================================
12 domains | ~118 equations | ~137 quantities | 15 laws
Goal: MULTI-PATH SOLVING — any target reachable via many routes.

DESIGN RULES
  1. Node SHARING: same physical object = ONE node (omega, T_period, f,
     m, t, g, v ... shared across domains). Shared nodes = path junctions.
  2. Redundant equations KEPT (v_avg two ways, eta two ways, P three
     producers, v_wave four producers) — each is another solving path.
  3. REVERSE edges included: info can flow backward through equations
     (know F,a -> find m). Signature: role bits & dir bits all zero.
  4. Semantic flags (16-28) exist because SI dims alone cannot separate
     torque/energy [1,2,-2], heat/work/U (path vs state), pressure/
     energy-density [-1,1,-2].

NODE FEATURES (32)
  0-6  SI exponents  [M, L, T, I, TH(temp), N(mol), J(lum)]
  7 rank  8 pseudo  9 const  10 angle  11 component  12 rate
  13 initial  14 avg  15 extremum
  16 force  17 energy  18 momentum  19 material_prop  20 geometric
  21 intensive  22 state_function  23 path_function  24 oscillatory
  25 wave  26 field  27 dimensionless_coef  28 flow   29-31 reserved

EQUATION FEATURES (32)
  0 linear  1 quadratic  2 trig  3 ratio  4 definition  5 sqrt
  6 exp/log  7 conservation  8-19 domain D1..D12  20 has_constant

LAW FEATURES (32)
  0 kinematic 1 conservation 2 force_law 3 empirical 4 violable
  5 sym_time 6 sym_space 7 sym_rotation 8 sym_none

EDGE FEATURES (16)
  ops: 0 MUL 1 ADD 2 TRIG 3 POW 4 DIV 5 PROP 6 EXPL(exp/log)
  role: 7 IN(qty->eq) 8 OUT(eq->qty) 9 BRIDGE
  dir : 10 FWD 11 BIDIR          (reverse edge = all role+dir zero)
  flag: 12 NEG 13 SQUARED 14 HAS_COEF(any numeric coeff) 15 CONJ
"""
import json, numpy as np
from collections import defaultdict, deque

NF, EFD = 32, 16

QK = dict(M=0,L=1,T=2,I=3,TH=4,N=5,J=6,rank=7,pseudo=8,const=9,angle=10,
          comp=11,rate=12,init=13,avg=14,ext=15,force=16,energy=17,mom=18,
          mat=19,geo=20,intens=21,state=22,path=23,osc=24,wave=25,field=26,
          coef=27,flow=28)
EQK = dict(lin=0,quad=1,trig=2,ratio=3,defn=4,sqrt=5,expl=6,cons=7,
           D1=8,D2=9,D3=10,D4=11,D5=12,D6=13,D7=14,D8=15,D9=16,D10=17,
           D11=18,D12=19,const=20,D13=21)
LK  = dict(kin=0,cons=1,force=2,emp=3,viol=4,sym_t=5,sym_s=6,sym_r=7,sym_n=8)
EK  = dict(MUL=0,ADD=1,TRIG=2,POW=3,DIV=4,PROP=5,EXPL=6,
           IN=7,OUT=8,BR=9,FWD=10,BIDIR=11,NEG=12,SQ=13,COEF=14,CONJ=15)

def feat(spec, keymap, dim=NF):
    v = [0.0]*dim
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok: continue
        if "=" in tok:
            k, val = tok.split("="); v[keymap[k]] = float(val)
        else:
            v[keymap[tok]] = 1.0
    return v

def espec(s):
    v = [0.0]*EFD
    for tok in s.split("+"):
        v[EK[tok.strip()]] = 1.0
    return v

# ════════════════════════════════════════════════════════════════════
# QUANTITIES  (id, feature-spec)   ~137 nodes
# ════════════════════════════════════════════════════════════════════
QUANTITIES = [
 # ── shared core ──────────────────────────────────────────────
 ("Q_t",      "T=1"),
 ("Q_m",      "M=1"),
 ("Q_g",      "L=1,T=-2,const,rate,field"),
 ("Q_theta",  "angle"),
 ("Q_s",      "L=1,rank"),
 ("Q_x",      "L=1,rank,comp"),
 ("Q_y",      "L=1,rank,comp"),
 ("Q_h",      "L=1,geo"),
 ("Q_r",      "L=1,geo"),
 ("Q_d",      "L=1,geo"),
 ("Q_u",      "L=1,T=-1,rank,rate,init"),
 ("Q_v",      "L=1,T=-1,rank,rate"),
 ("Q_v_avg",  "L=1,T=-1,rank,rate,avg"),
 ("Q_a",      "L=1,T=-2,rank,rate"),
 ("Q_F",      "M=1,L=1,T=-2,rank,force"),
 # ── D2 projectile ────────────────────────────────────────────
 ("Q_v0",     "L=1,T=-1,rate,init"),
 ("Q_ux",     "L=1,T=-1,rank,rate,comp,init"),
 ("Q_uy",     "L=1,T=-1,rank,rate,comp,init"),
 ("Q_vx",     "L=1,T=-1,rank,rate,comp"),
 ("Q_vy",     "L=1,T=-1,rank,rate,comp"),
 ("Q_R_range","L=1,ext"),
 ("Q_H_max",  "L=1,ext"),
 ("Q_T_flight","T=1,ext"),
 # ── D3 circular (omega/T_period/f SHARED with D9,D12) ────────
 ("Q_phi",    "angle"),
 ("Q_omega",  "T=-1,rank,pseudo,rate,osc"),
 ("Q_omega0", "T=-1,rank,pseudo,rate,init"),
 ("Q_alpha",  "T=-2,rank,pseudo,rate"),
 ("Q_v_tan",  "L=1,T=-1,rank,rate"),
 ("Q_ac",     "L=1,T=-2,rank,rate"),
 ("Q_a_tan",  "L=1,T=-2,rank,rate"),
 ("Q_a_tot",  "L=1,T=-2,rank,rate"),
 ("Q_T_period","T=1,osc"),
 ("Q_f",      "T=-1,rate,osc"),
 # ── D4 dynamics ──────────────────────────────────────────────
 ("Q_mu",     "coef"),
 ("Q_N_norm", "M=1,L=1,T=-2,rank,force"),
 ("Q_f_fric", "M=1,L=1,T=-2,rank,force"),
 ("Q_W_wt",   "M=1,L=1,T=-2,rank,force"),
 ("Q_N_incl", "M=1,L=1,T=-2,rank,force,comp"),
 ("Q_F_incl", "M=1,L=1,T=-2,rank,force,comp"),
 ("Q_p",      "M=1,L=1,T=-1,rank,mom"),
 ("Q_dp",     "M=1,L=1,T=-1,rank,mom"),
 ("Q_J_imp",  "M=1,L=1,T=-1,rank,mom"),
 # ── D5 energy ────────────────────────────────────────────────
 ("Q_KE",     "M=1,L=2,T=-2,energy,state"),
 ("Q_PE_g",   "M=1,L=2,T=-2,energy,state"),
 ("Q_PE_spr", "M=1,L=2,T=-2,energy,state"),
 ("Q_k_spr",  "M=1,T=-2,mat"),
 ("Q_x_spr",  "L=1,rank"),
 ("Q_F_spr",  "M=1,L=1,T=-2,rank,force"),
 ("Q_W_work", "M=1,L=2,T=-2,energy,path"),
 ("Q_E_mech", "M=1,L=2,T=-2,energy,state"),
 ("Q_P_pow",  "M=1,L=2,T=-3,rate"),
 ("Q_dKE",    "M=1,L=2,T=-2,energy"),
 ("Q_dE_mech","M=1,L=2,T=-2,energy"),
 # ── D6 collisions ────────────────────────────────────────────
 ("Q_m1",     "M=1"),
 ("Q_m2",     "M=1"),
 ("Q_v1i",    "L=1,T=-1,rank,rate,init"),
 ("Q_v2i",    "L=1,T=-1,rank,rate,init"),
 ("Q_v1f",    "L=1,T=-1,rank,rate"),
 ("Q_v2f",    "L=1,T=-1,rank,rate"),
 ("Q_p_tot",  "M=1,L=1,T=-1,rank,mom"),
 ("Q_v_cm",   "L=1,T=-1,rank,rate,avg"),
 ("Q_v_pi",   "L=1,T=-1,rank,rate"),
 ("Q_e_rest", "coef"),
 ("Q_KE_lost","M=1,L=2,T=-2,energy"),
 # ── D7 rotation ──────────────────────────────────────────────
 ("Q_tau",    "M=1,L=2,T=-2,rank,pseudo"),
 ("Q_I_rot",  "M=1,L=2,mat,geo"),
 ("Q_I_cm",   "M=1,L=2,mat,geo"),
 ("Q_d_axis", "L=1,geo"),
 ("Q_L_ang",  "M=1,L=2,T=-1,rank,pseudo,mom"),
 ("Q_dL",     "M=1,L=2,T=-1,rank,pseudo,mom"),
 ("Q_KE_rot", "M=1,L=2,T=-2,energy,state"),
 ("Q_KE_roll","M=1,L=2,T=-2,energy,state"),
 # ── D8 gravitation ───────────────────────────────────────────
 ("Q_G",      "M=-1,L=3,T=-2,const"),
 ("Q_M_big",  "M=1"),
 ("Q_r_orb",  "L=1,geo"),
 ("Q_F_grav", "M=1,L=1,T=-2,rank,force,field"),
 ("Q_g_field","L=1,T=-2,rank,rate,field"),
 ("Q_PE_orb", "M=1,L=2,T=-2,energy,state"),
 ("Q_v_orb",  "L=1,T=-1,rank,rate"),
 ("Q_T_orb",  "T=1"),
 ("Q_v_esc",  "L=1,T=-1,rate,ext"),
 ("Q_E_orb",  "M=1,L=2,T=-2,energy,state"),
 # ── D9 SHM (omega,T_period,f reused) ─────────────────────────
 ("Q_A_amp",  "L=1,osc,ext"),
 ("Q_phase",  "angle,osc"),
 ("Q_x_shm",  "L=1,rank,osc"),
 ("Q_v_shm",  "L=1,T=-1,rank,rate,osc"),
 ("Q_a_shm",  "L=1,T=-2,rank,rate,osc"),
 ("Q_L_pend", "L=1,geo"),
 ("Q_E_shm",  "M=1,L=2,T=-2,energy,state,osc"),
 ("Q_v_max",  "L=1,T=-1,rate,ext,osc"),
 ("Q_a_max",  "L=1,T=-2,rate,ext,osc"),
 # ── D10 fluids ───────────────────────────────────────────────
 ("Q_P_press","M=1,L=-1,T=-2,intens"),
 ("Q_P_atm",  "M=1,L=-1,T=-2,const,intens"),
 ("Q_rho",    "M=1,L=-3,mat,intens"),
 ("Q_V_vol",  "L=3,geo"),
 ("Q_A_area", "L=2,geo"),
 ("Q_v_fl",   "L=1,T=-1,rank,rate,flow"),
 ("Q_Q_flow", "L=3,T=-1,rate,flow"),
 ("Q_F_buoy", "M=1,L=1,T=-2,rank,force"),
 ("Q_E_bern", "M=1,L=-1,T=-2,energy,intens"),
 # ── D11 thermodynamics ───────────────────────────────────────
 ("Q_n_mol",  "N=1"),
 ("Q_R_gas",  "M=1,L=2,T=-2,TH=-1,N=-1,const"),
 ("Q_kB",     "M=1,L=2,T=-2,TH=-1,const"),
 ("Q_T_temp", "TH=1,intens,state"),
 ("Q_Th",     "TH=1,intens,state"),
 ("Q_Tc",     "TH=1,intens,state"),
 ("Q_dT",     "TH=1,intens"),
 ("Q_U_int",  "M=1,L=2,T=-2,energy,state"),
 ("Q_dU",     "M=1,L=2,T=-2,energy"),
 ("Q_Q_heat", "M=1,L=2,T=-2,energy,path"),
 ("Q_W_th",   "M=1,L=2,T=-2,energy,path"),
 ("Q_dV",     "L=3,geo"),
 ("Q_Vi",     "L=3,geo,init"),
 ("Q_Vf",     "L=3,geo"),
 ("Q_c_spec", "L=2,T=-2,TH=-1,mat"),
 ("Q_L_lat",  "L=2,T=-2,mat"),
 ("Q_eta",    "coef"),
 ("Q_dS",     "M=1,L=2,T=-2,TH=-1,state"),
 ("Q_v_rms",  "L=1,T=-1,rate,avg"),
 ("Q_KE_avg", "M=1,L=2,T=-2,energy,avg"),
 ("Q_M_molar","M=1,N=-1,mat"),
 ("Q_gamma",  "coef,mat"),
 # ── D12 waves (A_amp, omega, f, x, t reused) ─────────────────
 ("Q_lambda", "L=1,wave,geo"),
 ("Q_v_wave", "L=1,T=-1,rate,wave"),
 ("Q_T_tens", "M=1,L=1,T=-2,rank,force"),
 ("Q_mu_lin", "M=1,L=-1,mat"),
 ("Q_k_wave", "L=-1,wave"),
 ("Q_y_wave", "L=1,rank,wave,osc,comp"),
 ("Q_I_int",  "M=1,T=-3,intens,wave"),
 ("Q_f1",     "T=-1,rate,osc"),
 ("Q_f2",     "T=-1,rate,osc"),
 ("Q_f_beat", "T=-1,rate,osc"),
 ("Q_f_obs",  "T=-1,rate,osc"),
 ("Q_v_obs",  "L=1,T=-1,rank,rate"),
 ("Q_v_src",  "L=1,T=-1,rank,rate"),
 ("Q_L_str",  "L=1,geo"),
 ("Q_n_harm", "coef"),
 # ── D13 properties of matter ─────────────────────────────────
 ("Q_sigma_s","M=1,L=-1,T=-2,intens"),
 ("Q_eps_str","coef"),
 ("Q_Y_mod",  "M=1,L=-1,T=-2,mat,intens"),
 ("Q_dL_el",  "L=1,geo"),
 ("Q_L0",     "L=1,geo"),
 ("Q_U_el",   "M=1,L=2,T=-2,energy,state"),
 ("Q_eta_visc","M=1,L=-1,T=-1,mat"),
 ("Q_F_visc", "M=1,L=1,T=-2,rank,force"),
 ("Q_v_term", "L=1,T=-1,rank,rate,ext"),
 ("Q_rho_f",  "M=1,L=-3,mat,intens"),
 ("Q_S_surf", "M=1,T=-2,mat"),
 ("Q_h_cap",  "L=1,geo,ext"),
 ("Q_alpha_x","TH=-1,mat"),
 ("Q_k_therm","M=1,L=1,T=-3,TH=-1,mat"),
 ("Q_P_cond", "M=1,L=2,T=-3,rate,flow"),
 ("Q_sigma_SB","M=1,T=-3,TH=-4,const"),
 ("Q_P_rad",  "M=1,L=2,T=-3,rate"),
 # statics / conservation additions
 ("Q_x1",     "L=1,rank,comp"),
 ("Q_x2",     "L=1,rank,comp"),
 ("Q_x_cm",   "L=1,rank,avg"),
 ("Q_I2",     "M=1,L=2,mat,geo"),
 ("Q_om2",    "T=-1,rank,pseudo,rate"),
]

CONSTANTS = ["Q_g","Q_G","Q_R_gas","Q_kB","Q_P_atm","Q_sigma_SB"]

# ════════════════════════════════════════════════════════════════════
# LAWS  (id, feature-spec)
# ════════════════════════════════════════════════════════════════════
LAWS = [
 ("LAW_KIN_DEF",     "kin,sym_n"),
 ("LAW_NEWTON",      "force"),
 ("LAW_HOOKE",       "force,emp"),
 ("LAW_FRICTION",    "emp"),
 ("LAW_GRAVITATION", "force"),
 ("LAW_CONS_E",      "cons,sym_t"),
 ("LAW_CONS_P",      "cons,sym_s"),
 ("LAW_CONS_L",      "cons,sym_r"),
 ("LAW_ARCHIMEDES",  "emp"),
 ("LAW_CONTINUITY",  "cons"),
 ("LAW_BERNOULLI",   "cons,sym_t"),
 ("LAW_THERMO1",     "cons,sym_t"),
 ("LAW_THERMO2",     "emp"),
 ("LAW_IDEAL_GAS",   "emp"),
 ("LAW_WAVE",        "kin"),
]

# ════════════════════════════════════════════════════════════════════
# EQUATIONS: (id, domain, eq-feats, law, [(in_qty, edge-spec)...],
#             (out_qty, edge-spec))
# subtract = ADD+NEG | sqrt-output = POW on OUT | coeffs = COEF flag
# ════════════════════════════════════════════════════════════════════
EQS = [
# ── D1 linear kinematics (7) ─────────────────────────────────────
("EQ_v_uat",  "D1","lin","LAW_KIN_DEF",
  [("Q_u","ADD"),("Q_a","MUL"),("Q_t","MUL")],("Q_v","ADD")),
("EQ_s_ut2",  "D1","quad,const","LAW_KIN_DEF",
  [("Q_u","MUL"),("Q_t","MUL"),("Q_a","MUL+COEF")],("Q_s","ADD")),
("EQ_v2_2as", "D1","sqrt,quad,const","LAW_KIN_DEF",
  [("Q_u","POW+SQ"),("Q_a","MUL+COEF"),("Q_s","MUL")],("Q_v","POW")),
("EQ_s_uvt",  "D1","lin,const","LAW_KIN_DEF",
  [("Q_u","ADD+COEF"),("Q_v","ADD+COEF"),("Q_t","MUL")],("Q_s","MUL")),
("EQ_vavg_st","D1","ratio,defn","LAW_KIN_DEF",
  [("Q_s","DIV"),("Q_t","DIV")],("Q_v_avg","DIV")),
("EQ_s_vt2",  "D1","quad,const","LAW_KIN_DEF",
  [("Q_v","MUL"),("Q_t","MUL"),("Q_a","MUL+COEF+NEG")],("Q_s","ADD")),
("EQ_vavg_uv","D1","lin,const","LAW_KIN_DEF",
  [("Q_u","ADD+COEF"),("Q_v","ADD+COEF")],("Q_v_avg","ADD")),
# ── D2 projectile (10) ───────────────────────────────────────────
("EQ_ux",     "D2","trig","LAW_KIN_DEF",
  [("Q_v0","MUL"),("Q_theta","TRIG")],("Q_ux","MUL")),
("EQ_uy",     "D2","trig","LAW_KIN_DEF",
  [("Q_v0","MUL"),("Q_theta","TRIG")],("Q_uy","MUL")),
("EQ_vx_c",   "D2","defn,lin","LAW_KIN_DEF",
  [("Q_ux","PROP")],("Q_vx","PROP")),
("EQ_x_t",    "D2","lin","LAW_KIN_DEF",
  [("Q_ux","MUL"),("Q_t","MUL")],("Q_x","MUL")),
("EQ_y_t",    "D2","quad,const","LAW_KIN_DEF",
  [("Q_uy","MUL"),("Q_t","MUL"),("Q_g","MUL+NEG+COEF")],("Q_y","ADD")),
("EQ_vy_t",   "D2","lin","LAW_KIN_DEF",
  [("Q_uy","ADD"),("Q_g","MUL+NEG"),("Q_t","MUL")],("Q_vy","ADD")),
("EQ_range",  "D2","trig,quad","LAW_KIN_DEF",
  [("Q_v0","POW+SQ"),("Q_theta","TRIG"),("Q_g","DIV")],("Q_R_range","DIV")),
("EQ_hmax",   "D2","quad,ratio,const","LAW_KIN_DEF",
  [("Q_uy","POW+SQ"),("Q_g","DIV+COEF")],("Q_H_max","DIV")),
("EQ_tflight","D2","ratio,const","LAW_KIN_DEF",
  [("Q_uy","MUL+COEF"),("Q_g","DIV")],("Q_T_flight","DIV")),
("EQ_speed_xy","D2","sqrt,quad","LAW_KIN_DEF",
  [("Q_vx","POW+SQ"),("Q_vy","POW+SQ")],("Q_v","POW")),
# ── D3 circular kinematics (11) ──────────────────────────────────
("EQ_vtan",   "D3","lin","LAW_KIN_DEF",
  [("Q_omega","MUL"),("Q_r","MUL")],("Q_v_tan","MUL")),
("EQ_ac",     "D3","ratio,quad","LAW_KIN_DEF",
  [("Q_v_tan","POW+SQ"),("Q_r","DIV")],("Q_ac","DIV")),
("EQ_om_avg", "D3","ratio,defn","LAW_KIN_DEF",
  [("Q_phi","DIV"),("Q_t","DIV")],("Q_omega","DIV")),
("EQ_phi_t2", "D3","quad,const","LAW_KIN_DEF",
  [("Q_omega0","MUL"),("Q_t","MUL"),("Q_alpha","MUL+COEF")],("Q_phi","ADD")),
("EQ_om_lin", "D3","lin","LAW_KIN_DEF",
  [("Q_omega0","ADD"),("Q_alpha","MUL"),("Q_t","MUL")],("Q_omega","ADD")),
("EQ_om_sq",  "D3","sqrt,quad,const","LAW_KIN_DEF",
  [("Q_omega0","POW+SQ"),("Q_alpha","MUL+COEF"),("Q_phi","MUL")],("Q_omega","POW")),
("EQ_T_om",   "D3","ratio,defn,const","LAW_KIN_DEF",
  [("Q_omega","DIV")],("Q_T_period","DIV+COEF")),
("EQ_f_T",    "D3","ratio,defn","LAW_KIN_DEF",
  [("Q_T_period","DIV")],("Q_f","DIV")),
("EQ_om_f",   "D3","defn,lin,const","LAW_KIN_DEF",
  [("Q_f","MUL+COEF")],("Q_omega","MUL")),
("EQ_atan",   "D3","lin","LAW_KIN_DEF",
  [("Q_alpha","MUL"),("Q_r","MUL")],("Q_a_tan","MUL")),
("EQ_atot",   "D3","sqrt,quad","LAW_KIN_DEF",
  [("Q_ac","POW+SQ"),("Q_a_tan","POW+SQ")],("Q_a_tot","POW")),
# ── D4 dynamics (9) ──────────────────────────────────────────────
("EQ_F_ma",   "D4","lin","LAW_NEWTON",
  [("Q_m","MUL"),("Q_a","MUL")],("Q_F","MUL")),
("EQ_weight", "D4","lin","LAW_NEWTON",
  [("Q_m","MUL"),("Q_g","MUL")],("Q_W_wt","MUL")),
("EQ_fric",   "D4","lin","LAW_FRICTION",
  [("Q_mu","MUL"),("Q_N_norm","MUL")],("Q_f_fric","MUL")),
("EQ_p_mv",   "D4","lin,defn","LAW_NEWTON",
  [("Q_m","MUL"),("Q_v","MUL")],("Q_p","MUL")),
("EQ_F_dpt",  "D4","ratio","LAW_NEWTON",
  [("Q_dp","DIV"),("Q_t","DIV")],("Q_F","DIV")),
("EQ_J_Ft",   "D4","lin,defn","LAW_NEWTON",
  [("Q_F","MUL"),("Q_t","MUL")],("Q_J_imp","MUL")),
("EQ_J_dp",   "D4","defn,lin","LAW_NEWTON",
  [("Q_J_imp","PROP")],("Q_dp","PROP")),
("EQ_N_incl", "D4","trig","LAW_NEWTON",
  [("Q_m","MUL"),("Q_g","MUL"),("Q_theta","TRIG")],("Q_N_incl","MUL")),
("EQ_F_incl", "D4","trig","LAW_NEWTON",
  [("Q_m","MUL"),("Q_g","MUL"),("Q_theta","TRIG")],("Q_F_incl","MUL")),
# ── D5 work-energy-power (10) ────────────────────────────────────
("EQ_KE",     "D5","quad,const","LAW_CONS_E",
  [("Q_m","MUL+COEF"),("Q_v","POW+SQ")],("Q_KE","MUL")),
("EQ_PEg",    "D5","lin","LAW_CONS_E",
  [("Q_m","MUL"),("Q_g","MUL"),("Q_h","MUL")],("Q_PE_g","MUL")),
("EQ_PEspr",  "D5","quad,const","LAW_HOOKE",
  [("Q_k_spr","MUL+COEF"),("Q_x_spr","POW+SQ")],("Q_PE_spr","MUL")),
("EQ_Fspr",   "D5","lin","LAW_HOOKE",
  [("Q_k_spr","MUL+NEG"),("Q_x_spr","MUL")],("Q_F_spr","MUL")),
("EQ_W_Fd",   "D5","trig","LAW_NEWTON",
  [("Q_F","MUL"),("Q_d","MUL"),("Q_theta","TRIG")],("Q_W_work","MUL")),
("EQ_WE_thm", "D5","defn,cons","LAW_CONS_E",
  [("Q_W_work","PROP")],("Q_dKE","PROP")),
("EQ_Emech",  "D5","lin,cons","LAW_CONS_E",
  [("Q_KE","ADD"),("Q_PE_g","ADD")],("Q_E_mech","ADD")),
("EQ_P_Wt",   "D5","ratio,defn","LAW_KIN_DEF",
  [("Q_W_work","DIV"),("Q_t","DIV")],("Q_P_pow","DIV")),
("EQ_P_Fv",   "D5","lin","LAW_NEWTON",
  [("Q_F","MUL"),("Q_v","MUL")],("Q_P_pow","MUL")),
("EQ_dEm_Wnc","D5","defn,cons","LAW_CONS_E",
  [("Q_W_work","PROP")],("Q_dE_mech","PROP")),
# ── D6 collisions (8) — conservation as TWO producers of p_tot ──
("EQ_ptot_i", "D6","lin,cons","LAW_CONS_P",
  [("Q_m1","MUL"),("Q_v1i","MUL"),("Q_m2","MUL"),("Q_v2i","MUL")],("Q_p_tot","ADD")),
("EQ_ptot_f", "D6","lin,cons","LAW_CONS_P",
  [("Q_m1","MUL"),("Q_v1f","MUL"),("Q_m2","MUL"),("Q_v2f","MUL")],("Q_p_tot","ADD")),
("EQ_vcm",    "D6","ratio,cons","LAW_CONS_P",
  [("Q_m1","MUL"),("Q_v1i","MUL"),("Q_m2","MUL"),("Q_v2i","MUL")],("Q_v_cm","DIV")),
("EQ_v1f_el", "D6","ratio,cons","LAW_CONS_P",
  [("Q_m1","ADD"),("Q_m2","ADD"),("Q_v1i","MUL"),("Q_v2i","MUL")],("Q_v1f","DIV")),
("EQ_v2f_el", "D6","ratio,cons","LAW_CONS_P",
  [("Q_m1","ADD"),("Q_m2","ADD"),("Q_v1i","MUL"),("Q_v2i","MUL")],("Q_v2f","DIV")),
("EQ_v_pi",   "D6","ratio,cons","LAW_CONS_P",
  [("Q_m1","MUL"),("Q_v1i","MUL"),("Q_m2","MUL"),("Q_v2i","MUL")],("Q_v_pi","DIV")),
("EQ_e_rest", "D6","ratio,defn","LAW_KIN_DEF",
  [("Q_v2f","ADD"),("Q_v1f","ADD+NEG"),("Q_v1i","DIV"),("Q_v2i","DIV+NEG")],("Q_e_rest","DIV")),
("EQ_KElost", "D6","quad,ratio,const","LAW_CONS_E",
  [("Q_m1","MUL"),("Q_m2","MUL"),("Q_v1i","ADD"),("Q_v2i","ADD+NEG")],("Q_KE_lost","MUL")),
]

EQS += [
# ── D7 rotation (10) — roll_v outputs SHARED Q_v ─────────────────
("EQ_tau_rF", "D7","trig","LAW_NEWTON",
  [("Q_r","MUL"),("Q_F","MUL"),("Q_theta","TRIG")],("Q_tau","MUL")),
("EQ_tau_Ia", "D7","lin","LAW_NEWTON",
  [("Q_I_rot","MUL"),("Q_alpha","MUL")],("Q_tau","MUL")),
("EQ_I_mr2",  "D7","quad","LAW_KIN_DEF",
  [("Q_m","MUL"),("Q_r","POW+SQ")],("Q_I_rot","MUL")),
("EQ_L_Iw",   "D7","lin","LAW_CONS_L",
  [("Q_I_rot","MUL"),("Q_omega","MUL")],("Q_L_ang","MUL")),
("EQ_L_mvr",  "D7","lin","LAW_CONS_L",
  [("Q_m","MUL"),("Q_v","MUL"),("Q_r","MUL")],("Q_L_ang","MUL")),
("EQ_tau_dLt","D7","ratio","LAW_CONS_L",
  [("Q_dL","DIV"),("Q_t","DIV")],("Q_tau","DIV")),
("EQ_KErot",  "D7","quad,const","LAW_CONS_E",
  [("Q_I_rot","MUL+COEF"),("Q_omega","POW+SQ")],("Q_KE_rot","MUL")),
("EQ_KEroll", "D7","lin,cons","LAW_CONS_E",
  [("Q_KE","ADD"),("Q_KE_rot","ADD")],("Q_KE_roll","ADD")),
("EQ_parax",  "D7","quad","LAW_KIN_DEF",
  [("Q_I_cm","ADD"),("Q_m","MUL"),("Q_d_axis","POW+SQ")],("Q_I_rot","ADD")),
("EQ_roll_v", "D7","lin","LAW_KIN_DEF",
  [("Q_omega","MUL"),("Q_r","MUL")],("Q_v","MUL")),
# ── D8 gravitation (8) ───────────────────────────────────────────
("EQ_Fgrav",  "D8","ratio,quad","LAW_GRAVITATION",
  [("Q_G","MUL"),("Q_M_big","MUL"),("Q_m","MUL"),("Q_r_orb","DIV+SQ")],("Q_F_grav","DIV")),
("EQ_gfield", "D8","ratio,quad","LAW_GRAVITATION",
  [("Q_G","MUL"),("Q_M_big","MUL"),("Q_r_orb","DIV+SQ")],("Q_g_field","DIV")),
("EQ_PEorb",  "D8","ratio","LAW_GRAVITATION",
  [("Q_G","MUL+NEG"),("Q_M_big","MUL"),("Q_m","MUL"),("Q_r_orb","DIV")],("Q_PE_orb","DIV")),
("EQ_vorb",   "D8","sqrt,ratio","LAW_GRAVITATION",
  [("Q_G","MUL"),("Q_M_big","MUL"),("Q_r_orb","DIV")],("Q_v_orb","POW")),
("EQ_Torb",   "D8","sqrt,const","LAW_GRAVITATION",
  [("Q_r_orb","POW"),("Q_G","DIV"),("Q_M_big","DIV")],("Q_T_orb","POW+COEF")),
("EQ_vesc",   "D8","sqrt,const","LAW_GRAVITATION",
  [("Q_G","MUL+COEF"),("Q_M_big","MUL"),("Q_r_orb","DIV")],("Q_v_esc","POW")),
("EQ_F_mgf",  "D8","lin","LAW_GRAVITATION",
  [("Q_m","MUL"),("Q_g_field","MUL")],("Q_F_grav","MUL")),
("EQ_Eorb",   "D8","ratio,const","LAW_CONS_E",
  [("Q_G","MUL+NEG+COEF"),("Q_M_big","MUL"),("Q_m","MUL"),("Q_r_orb","DIV")],("Q_E_orb","DIV")),
# ── D9 SHM (12) — T_period gets 3rd producer here ────────────────
("EQ_xshm",   "D9","trig","LAW_HOOKE",
  [("Q_A_amp","MUL"),("Q_omega","TRIG"),("Q_t","TRIG"),("Q_phase","TRIG")],("Q_x_shm","MUL")),
("EQ_vshm_t", "D9","trig","LAW_HOOKE",
  [("Q_A_amp","MUL+NEG"),("Q_omega","MUL"),("Q_t","TRIG"),("Q_phase","TRIG")],("Q_v_shm","MUL")),
("EQ_ashm_t", "D9","trig,quad","LAW_HOOKE",
  [("Q_A_amp","MUL+NEG"),("Q_omega","POW+SQ"),("Q_t","TRIG"),("Q_phase","TRIG")],("Q_a_shm","MUL")),
("EQ_a_x",    "D9","lin,quad","LAW_HOOKE",
  [("Q_omega","POW+SQ"),("Q_x_shm","MUL+NEG")],("Q_a_shm","MUL")),
("EQ_om_km",  "D9","sqrt,ratio","LAW_HOOKE",
  [("Q_k_spr","MUL"),("Q_m","DIV")],("Q_omega","POW")),
("EQ_T_spr",  "D9","sqrt,const","LAW_HOOKE",
  [("Q_m","MUL"),("Q_k_spr","DIV")],("Q_T_period","POW+COEF")),
("EQ_T_pend", "D9","sqrt,const","LAW_KIN_DEF",
  [("Q_L_pend","MUL"),("Q_g","DIV")],("Q_T_period","POW+COEF")),
("EQ_E_kA",   "D9","quad,const","LAW_CONS_E",
  [("Q_k_spr","MUL+COEF"),("Q_A_amp","POW+SQ")],("Q_E_shm","MUL")),
("EQ_E_split","D9","quad,cons,const","LAW_CONS_E",
  [("Q_m","MUL+COEF"),("Q_v_shm","POW+SQ"),("Q_k_spr","MUL+COEF"),("Q_x_shm","POW+SQ")],("Q_E_shm","ADD")),
("EQ_vmax",   "D9","lin","LAW_HOOKE",
  [("Q_A_amp","MUL"),("Q_omega","MUL")],("Q_v_max","MUL")),
("EQ_amax",   "D9","quad","LAW_HOOKE",
  [("Q_A_amp","MUL"),("Q_omega","POW+SQ")],("Q_a_max","MUL")),
("EQ_v_Ax",   "D9","sqrt,quad","LAW_HOOKE",
  [("Q_omega","MUL"),("Q_A_amp","POW+SQ"),("Q_x_shm","POW+SQ+NEG")],("Q_v_shm","POW")),
# ── D10 fluids (8) — P_press gets 2 producers, m gets 2nd ────────
("EQ_P_FA",   "D10","ratio,defn","LAW_NEWTON",
  [("Q_F","DIV"),("Q_A_area","DIV")],("Q_P_press","DIV")),
("EQ_rho_mV", "D10","ratio,defn","LAW_KIN_DEF",
  [("Q_m","DIV"),("Q_V_vol","DIV")],("Q_rho","DIV")),
("EQ_m_rhoV", "D10","lin","LAW_KIN_DEF",
  [("Q_rho","MUL"),("Q_V_vol","MUL")],("Q_m","MUL")),
("EQ_P_hydro","D10","lin","LAW_BERNOULLI",
  [("Q_P_atm","ADD"),("Q_rho","MUL"),("Q_g","MUL"),("Q_h","MUL")],("Q_P_press","ADD")),
("EQ_buoy",   "D10","lin","LAW_ARCHIMEDES",
  [("Q_rho","MUL"),("Q_V_vol","MUL"),("Q_g","MUL")],("Q_F_buoy","MUL")),
("EQ_Qflow",  "D10","lin,defn","LAW_CONTINUITY",
  [("Q_A_area","MUL"),("Q_v_fl","MUL")],("Q_Q_flow","MUL")),
("EQ_bern",   "D10","quad,cons,const","LAW_BERNOULLI",
  [("Q_P_press","ADD"),("Q_rho","MUL+COEF"),("Q_v_fl","POW+SQ"),("Q_g","MUL"),("Q_h","MUL")],("Q_E_bern","ADD")),
("EQ_torr",   "D10","sqrt,const","LAW_BERNOULLI",
  [("Q_g","MUL+COEF"),("Q_h","MUL")],("Q_v_fl","POW")),
# ── D11 thermodynamics (13) — P 3rd producer, eta 2, dU 2, W_th 2, Q 2
("EQ_gas",    "D11","ratio,cons","LAW_IDEAL_GAS",
  [("Q_n_mol","MUL"),("Q_R_gas","MUL"),("Q_T_temp","MUL"),("Q_V_vol","DIV")],("Q_P_press","DIV")),
("EQ_1law",   "D11","lin,cons","LAW_THERMO1",
  [("Q_Q_heat","ADD"),("Q_W_th","ADD+NEG")],("Q_dU","ADD")),
("EQ_W_PdV",  "D11","lin","LAW_THERMO1",
  [("Q_P_press","MUL"),("Q_dV","MUL")],("Q_W_th","MUL")),
("EQ_Q_mcdT", "D11","lin","LAW_THERMO1",
  [("Q_m","MUL"),("Q_c_spec","MUL"),("Q_dT","MUL")],("Q_Q_heat","MUL")),
("EQ_Q_mL",   "D11","lin","LAW_THERMO1",
  [("Q_m","MUL"),("Q_L_lat","MUL")],("Q_Q_heat","MUL")),
("EQ_eta_WQ", "D11","ratio,defn","LAW_THERMO2",
  [("Q_W_th","DIV"),("Q_Q_heat","DIV")],("Q_eta","DIV")),
("EQ_carnot", "D11","ratio","LAW_THERMO2",
  [("Q_Tc","DIV+NEG"),("Q_Th","DIV")],("Q_eta","ADD")),
("EQ_dS",     "D11","ratio,defn","LAW_THERMO2",
  [("Q_Q_heat","DIV"),("Q_T_temp","DIV")],("Q_dS","DIV")),
("EQ_U_nRT",  "D11","lin,const","LAW_IDEAL_GAS",
  [("Q_n_mol","MUL+COEF"),("Q_R_gas","MUL"),("Q_T_temp","MUL")],("Q_U_int","MUL")),
("EQ_dU_nRdT","D11","lin,const","LAW_IDEAL_GAS",
  [("Q_n_mol","MUL+COEF"),("Q_R_gas","MUL"),("Q_dT","MUL")],("Q_dU","MUL")),
("EQ_vrms",   "D11","sqrt,const","LAW_IDEAL_GAS",
  [("Q_R_gas","MUL+COEF"),("Q_T_temp","MUL"),("Q_M_molar","DIV")],("Q_v_rms","POW")),
("EQ_KEavg",  "D11","lin,const","LAW_IDEAL_GAS",
  [("Q_kB","MUL+COEF"),("Q_T_temp","MUL")],("Q_KE_avg","MUL")),
("EQ_W_iso",  "D11","expl","LAW_THERMO1",
  [("Q_n_mol","MUL"),("Q_R_gas","MUL"),("Q_T_temp","MUL"),("Q_Vf","EXPL"),("Q_Vi","EXPL+NEG")],("Q_W_th","MUL")),
# ── D12 waves (12) — v_wave 4 producers, f 3rd, lambda 2nd ───────
("EQ_v_flam", "D12","lin","LAW_WAVE",
  [("Q_f","MUL"),("Q_lambda","MUL")],("Q_v_wave","MUL")),
("EQ_v_str",  "D12","sqrt,ratio","LAW_WAVE",
  [("Q_T_tens","MUL"),("Q_mu_lin","DIV")],("Q_v_wave","POW")),
("EQ_kwave",  "D12","ratio,const","LAW_WAVE",
  [("Q_lambda","DIV")],("Q_k_wave","DIV+COEF")),
("EQ_ywave",  "D12","trig","LAW_WAVE",
  [("Q_A_amp","MUL"),("Q_k_wave","TRIG"),("Q_x","TRIG"),("Q_omega","TRIG"),("Q_t","TRIG")],("Q_y_wave","MUL")),
("EQ_I_PA",   "D12","ratio,defn","LAW_WAVE",
  [("Q_P_pow","DIV"),("Q_A_area","DIV")],("Q_I_int","DIV")),
("EQ_I_A2",   "D12","quad","LAW_WAVE",
  [("Q_A_amp","PROP+SQ")],("Q_I_int","PROP")),
("EQ_beat",   "D12","lin","LAW_WAVE",
  [("Q_f1","ADD"),("Q_f2","ADD+NEG")],("Q_f_beat","ADD")),
("EQ_doppler","D12","ratio","LAW_WAVE",
  [("Q_f","MUL"),("Q_v_wave","PROP"),("Q_v_obs","ADD"),("Q_v_src","DIV+NEG")],("Q_f_obs","MUL")),
("EQ_stand",  "D12","ratio,const","LAW_WAVE",
  [("Q_L_str","MUL+COEF"),("Q_n_harm","DIV")],("Q_lambda","DIV")),
("EQ_f_fund", "D12","ratio,const","LAW_WAVE",
  [("Q_v_wave","MUL"),("Q_L_str","DIV+COEF")],("Q_f","DIV")),
("EQ_v_sound","D12","sqrt","LAW_IDEAL_GAS",
  [("Q_gamma","MUL"),("Q_R_gas","MUL"),("Q_T_temp","MUL"),("Q_M_molar","DIV")],("Q_v_wave","POW")),
("EQ_v_omk",  "D12","ratio","LAW_WAVE",
  [("Q_omega","DIV"),("Q_k_wave","DIV")],("Q_v_wave","DIV")),
# ── D13 properties of matter (11) ────────────────────────────────
("EQ_stress", "D13","ratio,defn","LAW_HOOKE",
  [("Q_F","DIV"),("Q_A_area","DIV")],("Q_sigma_s","DIV")),
("EQ_strain", "D13","ratio,defn","LAW_HOOKE",
  [("Q_dL_el","DIV"),("Q_L0","DIV")],("Q_eps_str","DIV")),
("EQ_young",  "D13","ratio,defn","LAW_HOOKE",
  [("Q_sigma_s","DIV"),("Q_eps_str","DIV")],("Q_Y_mod","DIV")),
("EQ_U_el",   "D13","lin,const","LAW_HOOKE",
  [("Q_F","MUL+COEF"),("Q_dL_el","MUL")],("Q_U_el","MUL")),
("EQ_stokes", "D13","lin,const","LAW_FRICTION",
  [("Q_eta_visc","MUL+COEF"),("Q_r","MUL"),("Q_v","MUL")],("Q_F_visc","MUL")),
("EQ_vterm",  "D13","quad,ratio,const","LAW_FRICTION",
  [("Q_r","POW+SQ"),("Q_rho","ADD"),("Q_rho_f","ADD+NEG"),
   ("Q_g","MUL"),("Q_eta_visc","DIV")],("Q_v_term","DIV+COEF")),
("EQ_dP_surf","D13","ratio,const","LAW_BERNOULLI",
  [("Q_S_surf","MUL+COEF"),("Q_r","DIV")],("Q_P_press","DIV")),
("EQ_capil",  "D13","trig,ratio,const","LAW_BERNOULLI",
  [("Q_S_surf","MUL+COEF"),("Q_theta","TRIG"),("Q_rho","DIV"),
   ("Q_g","DIV"),("Q_r","DIV")],("Q_h_cap","DIV")),
("EQ_thermx", "D13","lin","LAW_IDEAL_GAS",
  [("Q_alpha_x","MUL"),("Q_L0","MUL"),("Q_dT","MUL")],("Q_dL_el","MUL")),
("EQ_conduct","D13","ratio","LAW_THERMO2",
  [("Q_k_therm","MUL"),("Q_A_area","MUL"),("Q_dT","MUL"),
   ("Q_L0","DIV")],("Q_P_cond","DIV")),
("EQ_stefan", "D13","quad,const","LAW_THERMO2",
  [("Q_sigma_SB","MUL"),("Q_A_area","MUL"),("Q_T_temp","POW")],
  ("Q_P_rad","MUL")),
# ── statics + explicit L conservation ────────────────────────────
("EQ_xcm",    "D6","ratio,cons","LAW_CONS_P",
  [("Q_m1","MUL"),("Q_x1","MUL"),("Q_m2","MUL"),("Q_x2","MUL")],
  ("Q_x_cm","DIV")),
("EQ_L_after","D7","lin,cons","LAW_CONS_L",
  [("Q_I2","MUL"),("Q_om2","MUL")],("Q_L_ang","MUL")),
]

# ════════════════════════════════════════════════════════════════════
# BRIDGES — same concept, different context (bidirectional PROP)
# ════════════════════════════════════════════════════════════════════
BRIDGES = [
 ("Q_v","Q_v_tan"),("Q_v","Q_v_shm"),("Q_v","Q_v_fl"),("Q_v","Q_v_wave"),
 ("Q_v","Q_vy"),("Q_v","Q_v_cm"),("Q_v","Q_v_orb"),("Q_v","Q_v_rms"),
 ("Q_u","Q_ux"),("Q_u","Q_uy"),("Q_u","Q_v0"),("Q_u","Q_omega0"),
 ("Q_s","Q_x"),("Q_s","Q_y"),("Q_s","Q_phi"),("Q_s","Q_x_shm"),
 ("Q_x_spr","Q_x_shm"),("Q_x_spr","Q_s"),
 ("Q_a","Q_a_tan"),("Q_a","Q_ac"),("Q_a","Q_a_shm"),("Q_a","Q_alpha"),
 ("Q_a","Q_g_field"),
 ("Q_v","Q_omega"),("Q_theta","Q_phi"),("Q_theta","Q_phase"),
 ("Q_F","Q_W_wt"),("Q_F","Q_f_fric"),("Q_F","Q_F_spr"),("Q_F","Q_F_grav"),
 ("Q_F","Q_F_buoy"),("Q_F","Q_N_norm"),("Q_F","Q_T_tens"),
 ("Q_p","Q_L_ang"),("Q_p","Q_p_tot"),("Q_p","Q_dp"),
 ("Q_KE","Q_KE_rot"),("Q_KE","Q_KE_avg"),
 ("Q_E_mech","Q_E_shm"),("Q_E_mech","Q_E_orb"),("Q_E_mech","Q_U_int"),
 ("Q_PE_g","Q_PE_orb"),("Q_PE_g","Q_PE_spr"),
 ("Q_W_work","Q_W_th"),("Q_W_work","Q_Q_heat"),
 ("Q_m","Q_m1"),("Q_m","Q_m2"),("Q_m","Q_M_big"),
 ("Q_r","Q_r_orb"),("Q_r","Q_d_axis"),("Q_r","Q_L_pend"),("Q_h","Q_d"),
 ("Q_f","Q_f1"),("Q_f","Q_f2"),("Q_f","Q_f_obs"),
 ("Q_rho","Q_mu_lin"),
 ("Q_sigma_s","Q_P_press"),("Q_F_visc","Q_F"),("Q_v_term","Q_v"),
 ("Q_U_el","Q_PE_spr"),("Q_h_cap","Q_h"),("Q_P_cond","Q_P_pow"),
 ("Q_P_rad","Q_P_pow"),("Q_rho_f","Q_rho"),("Q_dL_el","Q_x_spr"),
 ("Q_x_cm","Q_x"),("Q_x1","Q_x"),("Q_x2","Q_x"),
 ("Q_om2","Q_omega"),("Q_I2","Q_I_rot"),("Q_L0","Q_L_str"),
]

# ════════════════════════════════════════════════════════════════════
# BUILDER
# ════════════════════════════════════════════════════════════════════
def build():
    X, meta, id2idx = [], [], {}

    def add_node(nid, ntype, fv):
        assert nid not in id2idx, f"duplicate node {nid}"
        id2idx[nid] = len(X)
        X.append(fv)
        meta.append({"index": len(meta), "id": nid, "node_type": ntype})

    for lid, spec in LAWS:
        add_node(lid, "law", feat(spec, LK))
    for qid, spec in QUANTITIES:
        add_node(qid, "quantity", feat(spec, QK))
    for eid, dom, espec_s, law, ins, out in EQS:
        add_node(eid, "equation", feat(espec_s + "," + dom, EQK))

    edges, E, emeta = [], [], []

    def add_edge(src, dst, fv, kind):
        edges.append([id2idx[src], id2idx[dst]])
        E.append(fv)
        emeta.append({"src": src, "dst": dst, "kind": kind})

    # forward physics edges
    for eid, dom, _, law, ins, (oq, ospec) in EQS:
        for q, spec in ins:
            assert q in id2idx, f"{eid}: unknown input {q}"
            fv = espec(spec); fv[EK["IN"]] = 1.0; fv[EK["FWD"]] = 1.0
            add_edge(q, eid, fv, "qty_in")
        assert oq in id2idx, f"{eid}: unknown output {oq}"
        fv = espec(ospec); fv[EK["OUT"]] = 1.0; fv[EK["FWD"]] = 1.0
        add_edge(eid, oq, fv, "qty_out")
        assert law in id2idx, f"{eid}: unknown law {law}"
        fv = espec("PROP"); fv[EK["FWD"]] = 1.0
        add_edge(law, eid, fv, "law_gen")

    # bridges (bidirectional)
    for a, b in BRIDGES:
        assert a in id2idx and b in id2idx, f"bad bridge {a}-{b}"
        for s, dd in ((a, b), (b, a)):
            fv = espec("PROP"); fv[EK["BR"]] = 1.0; fv[EK["BIDIR"]] = 1.0
            add_edge(s, dd, fv, "bridge")

    # REVERSE edges: for every qty_in / qty_out edge add flipped copy.
    # Signature: op+NEG/SQ/COEF flags KEPT, role bits & dir bits ZERO.
    n_fwd = len(edges)
    for i in range(n_fwd):
        if emeta[i]["kind"] not in ("qty_in", "qty_out"):
            continue
        s, dd = edges[i]
        fv = list(E[i])
        for k in ("IN", "OUT", "BR", "FWD", "BIDIR"):
            fv[EK[k]] = 0.0
        edges.append([dd, s]); E.append(fv)
        emeta.append({"src": emeta[i]["dst"], "dst": emeta[i]["src"],
                      "kind": "reverse"})

    return (np.array(X, dtype=np.float32), np.array(edges, dtype=np.int64),
            np.array(E, dtype=np.float32), meta, emeta, id2idx)


def verify(X, edges, E, meta, emeta, id2idx):
    ok = True
    # every equation: >=1 input, exactly 1 output
    ins_of  = defaultdict(int); outs_of = defaultdict(int)
    for em in emeta:
        if em["kind"] == "qty_in":  ins_of[em["dst"]]  += 1
        if em["kind"] == "qty_out": outs_of[em["src"]] += 1
    for eid, *_ in EQS:
        if ins_of[eid] < 1:  print(f"  FAIL {eid}: no inputs");  ok = False
        if outs_of[eid] != 1: print(f"  FAIL {eid}: outs={outs_of[eid]}"); ok = False

    # connectivity (undirected BFS from Q_m)
    adj = defaultdict(set)
    for s, d in edges:
        adj[s].add(d); adj[d].add(s)
    seen, dq = {id2idx["Q_m"]}, deque([id2idx["Q_m"]])
    while dq:
        n = dq.popleft()
        for nb in adj[n]:
            if nb not in seen:
                seen.add(nb); dq.append(nb)
    conn = len(seen) / len(X) * 100
    if conn < 100:
        missing = [m["id"] for m in meta if m["index"] not in seen]
        print(f"  WARN connectivity {conn:.1f}% — unreachable: {missing}")
    return ok, conn


def stats(X, edges, E, meta, emeta, id2idx):
    by_type = defaultdict(int)
    for m in meta: by_type[m["node_type"]] += 1
    by_kind = defaultdict(int)
    for em in emeta: by_kind[em["kind"]] += 1

    producers = defaultdict(list)   # qty -> [eq,...]
    consumers = defaultdict(int)
    for em in emeta:
        if em["kind"] == "qty_out": producers[em["dst"]].append(em["src"])
        if em["kind"] == "qty_in":  consumers[em["src"]] += 1

    print(f"\nNODES {len(X)}: " +
          ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())))
    print(f"EDGES {len(edges)}: " +
          ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items())))

    multi = sorted(((q, p) for q, p in producers.items() if len(p) >= 2),
                   key=lambda x: -len(x[1]))
    print(f"\nMULTI-PATH DENSITY — quantities with 2+ producing equations "
          f"({len(multi)} total):")
    for q, p in multi:
        print(f"  {q:12s} <- {len(p)} equations: {', '.join(p)}")

    hubs = sorted(consumers.items(), key=lambda x: -x[1])[:10]
    print(f"\nTOP JUNCTION NODES (used as input by most equations):")
    for q, c in hubs:
        print(f"  {q:12s} feeds {c} equations")


def save(X, edges, E, meta, emeta, path):
    doc = {
        "version": "cm_fullgraph_v1.0",
        "goal": "multi-path solving across all classical mechanics",
        "node_feature_dim": NF, "edge_feature_dim": EFD,
        "node_schema": {v: k for k, v in QK.items()},
        "eq_schema":   {v: k for k, v in EQK.items()},
        "law_schema":  {v: k for k, v in LK.items()},
        "edge_schema": {v: k for k, v in EK.items()},
        "reverse_edge_rule": "role bits (IN/OUT/BR) and dir bits (FWD/BIDIR) all zero",
        "constants": CONSTANTS,
        "domains": ["D1 linear kinematics","D2 projectile","D3 circular",
                    "D4 dynamics","D5 work-energy-power","D6 collisions",
                    "D7 rotation","D8 gravitation","D9 SHM","D10 fluids",
                    "D11 thermodynamics","D12 waves"],
        "X": X.tolist(), "edge_index": edges.tolist(), "E": E.tolist(),
        "node_metadata": meta, "edge_metadata": emeta,
    }
    with open(path, "w") as f:
        json.dump(doc, f)
    import os
    print(f"\nSaved {path}  ({os.path.getsize(path)/1024:.0f} KB)")


if __name__ == "__main__":
    print("="*66)
    print("BUILDING FULL CLASSICAL MECHANICS GRAPH")
    print("="*66)
    X, edges, E, meta, emeta, id2idx = build()
    ok, conn = verify(X, edges, E, meta, emeta, id2idx)
    stats(X, edges, E, meta, emeta, id2idx)
    print(f"\nVERIFY: {'ALL PASS' if ok else 'FAILURES ABOVE'}  |  "
          f"connectivity from Q_m: {conn:.1f}%")
    save(X, edges, E, meta, emeta, "cm_fullgraph_v1.json")
    print("="*66)
