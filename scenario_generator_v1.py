"""
SCENARIO DATA GENERATOR v1 — consistent physical worlds
========================================================
One sample = one coherent physics scenario. Every quantity in the
scenario satisfies EVERY graph equation that touches it — including
multi-producer identities (T_period 3 ways, P_press 3 ways, eta 2 ways,
W_th 2 ways, E_shm 2 ways, v_wave chain). Chains therefore EXIST in the
data and the GNN can learn to route.

NORMALIZATION (v8 scheme — preserves multiplicative structure):
    mag  = log10(|x| + 1e-12) / 10        sign = +1 / -1 (0 if absent)
    =>  F = m*a  becomes  mag_F = mag_m + mag_a   EXACTLY (linear!)

CONSTANTS always revealed at train time: g, G, R_gas, kB, P_atm.

BANNED COMBOS (ladder test): listed input-set->target pairs are never
used as (revealed ⊆ inputs∪constants, target) during training; a fresh
TEST set with exactly those reveals is emitted for honest evaluation.

OUTPUT
  scenario_data_v1.npz   mags float32 [N,Q] | signs int8 | exist bool
                         scen_id int16
  scenario_test_v1.npz   per-ban test arrays
  scenario_meta_v1.json  qty order, norm spec, scenarios, bans, consts

RUN:  python scenario_generator_v1.py [N_PER_SCENARIO]   (default 5000;
      use 50000 on Colab for full training set)
"""
import json, sys, math
import numpy as np

rng = np.random.default_rng(42)
EPS = 1e-12

# ── graph quantity order (columns MUST match graph) ─────────────────
G = json.load(open("cm_fullgraph_v1.json"))
QTY = [m["id"] for m in G["node_metadata"] if m["node_type"] == "quantity"]
QI  = {q: i for i, q in enumerate(QTY)}
NQ  = len(QTY)

CONST_VALS = {"Q_g": 9.81, "Q_G": 6.674e-11, "Q_R_gas": 8.314,
              "Q_kB": 1.381e-23, "Q_P_atm": 101325.0, "Q_sigma_SB": 5.670e-8}

def mag(x):  return math.log10(abs(x) + EPS) / 10.0
def sgn(x):  return 0 if x == 0 else (1 if x > 0 else -1)
def LU(a, b):        # log-uniform positive
    return float(np.exp(rng.uniform(np.log(a), np.log(b))))
def U(a, b):  return float(rng.uniform(a, b))

# ════════════════════════════════════════════════════════════════════
# SCENARIOS — each returns {qty_id: raw_value}, all equations exact
# ════════════════════════════════════════════════════════════════════
def s1_linear():
    u, a, t = LU(.1, 50), LU(.1, 20), LU(.1, 30)
    v = u + a*t; s = u*t + .5*a*t*t
    return {"Q_u":u,"Q_a":a,"Q_t":t,"Q_v":v,"Q_s":s,"Q_v_avg":(u+v)/2}

def s2_projectile():
    v0, th = LU(5, 80), U(math.radians(10), math.radians(80))
    g = CONST_VALS["Q_g"]
    ux, uy = v0*math.cos(th), v0*math.sin(th)
    Tf = 2*uy/g; t = U(.05, .95)*Tf
    vx, vy = ux, uy - g*t
    x, y   = ux*t, uy*t - .5*g*t*t
    return {"Q_v0":v0,"Q_theta":th,"Q_g":g,"Q_ux":ux,"Q_uy":uy,
            "Q_vx":vx,"Q_vy":vy,"Q_x":x,"Q_y":y,"Q_t":t,
            "Q_R_range":v0*v0*math.sin(2*th)/g,"Q_H_max":uy*uy/(2*g),
            "Q_T_flight":Tf,"Q_v":math.hypot(vx,vy)}

def s3_circular():
    om0, al, t, r = LU(.1,20), LU(.05,10), LU(.1,20), LU(.05,10)
    om  = om0 + al*t
    phi = om0*t + .5*al*t*t
    vt, ac, at = om*r, (om*r)**2/r, al*r
    return {"Q_omega0":om0,"Q_alpha":al,"Q_t":t,"Q_r":r,"Q_omega":om,
            "Q_phi":phi,"Q_v_tan":vt,"Q_ac":ac,"Q_a_tan":at,
            "Q_a_tot":math.hypot(ac,at),
            "Q_T_period":2*math.pi/om,"Q_f":om/(2*math.pi)}

def s4_incline():
    g = CONST_VALS["Q_g"]
    while True:
        th, mu = U(math.radians(15), math.radians(70)), LU(.05,.8)
        if math.sin(th) > mu*math.cos(th) + .02: break
    m, u, t = LU(.5,500), LU(.1,20), LU(.1,15)
    a  = g*(math.sin(th) - mu*math.cos(th))
    v  = u + a*t; s = u*t + .5*a*t*t
    N  = m*g*math.cos(th); F = m*a
    return {"Q_theta":th,"Q_mu":mu,"Q_m":m,"Q_u":u,"Q_t":t,"Q_g":g,
            "Q_a":a,"Q_v":v,"Q_s":s,"Q_v_avg":(u+v)/2,
            "Q_N_norm":N,"Q_N_incl":N,"Q_F_incl":m*g*math.sin(th),
            "Q_f_fric":mu*N,"Q_W_wt":m*g,"Q_F":F,"Q_p":m*v,
            "Q_dp":m*(v-u),"Q_J_imp":F*t}

def s5_energy():
    g = CONST_VALS["Q_g"]
    m, v, h  = LU(.5,500), LU(.5,40), LU(.2,100)
    k, xs    = LU(5,5000), LU(.01,2)
    F, d, th = LU(1,2000), LU(.1,50), U(math.radians(5),math.radians(75))
    W = F*d*math.cos(th); P = F*v; t = W/P
    KE, PE = .5*m*v*v, m*g*h
    return {"Q_m":m,"Q_v":v,"Q_h":h,"Q_g":g,"Q_k_spr":k,"Q_x_spr":xs,
            "Q_F":F,"Q_d":d,"Q_theta":th,"Q_t":t,
            "Q_KE":KE,"Q_PE_g":PE,"Q_E_mech":KE+PE,
            "Q_PE_spr":.5*k*xs*xs,"Q_F_spr":-k*xs,
            "Q_W_work":W,"Q_P_pow":P,"Q_dKE":W,"Q_dE_mech":W}

def s6_collision():
    m1, m2 = LU(.2,200), LU(.2,200)
    v1i = U(1,30); v2i = U(-20,20); e = U(0,1)
    if abs(v1i-v2i) < .5: v2i = v1i - 2.0
    M = m1+m2
    vcm = (m1*v1i+m2*v2i)/M
    v1f = vcm + m2/M*e*(v2i-v1i)
    v2f = vcm + m1/M*e*(v1i-v2i)
    KEl = .5*m1*v1i**2+.5*m2*v2i**2 - (.5*m1*v1f**2+.5*m2*v2f**2)
    x1, x2 = U(-10,10), U(-10,10)
    return {"Q_m1":m1,"Q_m2":m2,"Q_v1i":v1i,"Q_v2i":v2i,"Q_e_rest":e,
            "Q_v1f":v1f,"Q_v2f":v2f,"Q_p_tot":m1*v1i+m2*v2i,
            "Q_v_cm":vcm,"Q_v_pi":vcm,"Q_KE_lost":max(KEl,EPS),
            "Q_x1":x1,"Q_x2":x2,"Q_x_cm":(m1*x1+m2*x2)/M}

def s7_rotation():
    m, Icm, dax = LU(.5,300), LU(.01,50), LU(.05,5)
    om, al, t   = LU(.2,30), LU(.05,10), LU(.1,15)
    th          = U(math.radians(20), math.radians(90))
    I  = Icm + m*dax*dax
    I2 = I*LU(.3,3.0)                # after pulling arms in/out
    r  = math.sqrt(I/m)              # => I = m r^2 also exact
    tau= I*al; v = om*r
    F  = tau/(r*math.sin(th))
    KE, KEr = .5*m*v*v, .5*I*om*om
    return {"Q_m":m,"Q_I_cm":Icm,"Q_d_axis":dax,"Q_omega":om,
            "Q_alpha":al,"Q_t":t,"Q_theta":th,"Q_I_rot":I,"Q_r":r,
            "Q_tau":tau,"Q_dL":tau*t,"Q_L_ang":I*om,"Q_v":v,"Q_F":F,
            "Q_KE":KE,"Q_KE_rot":KEr,"Q_KE_roll":KE+KEr,
            "Q_T_period":2*math.pi/om,"Q_f":om/(2*math.pi),
            "Q_I2":I2,"Q_om2":I*om/I2}

def s8_orbit():
    Gc = CONST_VALS["Q_G"]
    M, m, r = LU(1e22,1e30), LU(1,1e5), LU(1e6,1e11)
    gf = Gc*M/r**2
    return {"Q_G":Gc,"Q_M_big":M,"Q_m":m,"Q_r_orb":r,
            "Q_F_grav":Gc*M*m/r**2,"Q_g_field":gf,
            "Q_PE_orb":-Gc*M*m/r,"Q_v_orb":math.sqrt(Gc*M/r),
            "Q_T_orb":2*math.pi*math.sqrt(r**3/(Gc*M)),
            "Q_v_esc":math.sqrt(2*Gc*M/r),"Q_E_orb":-Gc*M*m/(2*r)}

def s9_shm():
    g = CONST_VALS["Q_g"]
    m, k, A = LU(.1,100), LU(1,2000), LU(.02,3)
    ph, t   = U(0,2*math.pi), LU(.05,20)
    om = math.sqrt(k/m); T = 2*math.pi/om
    arg = om*t + ph
    x, v = A*math.cos(arg), -A*om*math.sin(arg)
    return {"Q_m":m,"Q_k_spr":k,"Q_A_amp":A,"Q_phase":ph,"Q_t":t,
            "Q_g":g,"Q_omega":om,"Q_T_period":T,"Q_f":1/T,
            "Q_L_pend":g*(T/(2*math.pi))**2,      # pendulum consistent
            "Q_x_shm":x,"Q_v_shm":v,"Q_a_shm":-om*om*x,
            "Q_E_shm":.5*k*A*A,"Q_v_max":A*om,"Q_a_max":A*om*om}

def s10_fluid():
    g, Pa = CONST_VALS["Q_g"], CONST_VALS["Q_P_atm"]
    rho, V, h, Aa = LU(500,13600), LU(.001,10), LU(.1,50), LU(1e-4,1)
    vfl = math.sqrt(2*g*h)                       # Torricelli exact
    P   = Pa + rho*g*h
    m   = rho*V
    return {"Q_rho":rho,"Q_V_vol":V,"Q_h":h,"Q_A_area":Aa,"Q_g":g,
            "Q_P_atm":Pa,"Q_m":m,"Q_P_press":P,"Q_F":P*Aa,
            "Q_F_buoy":rho*V*g,"Q_v_fl":vfl,"Q_Q_flow":Aa*vfl,
            "Q_E_bern":P + .5*rho*vfl*vfl + rho*g*h}

def s11_thermo():
    R, kB = CONST_VALS["Q_R_gas"], CONST_VALS["Q_kB"]
    n, T, V   = LU(.1,100), LU(150,1500), LU(.001,5)
    m, dT     = LU(.05,50), LU(1,300)
    Mm        = LU(.002,.2)
    x         = U(.1,4.0)                    # ln(Vf/Vi), bounded
    P  = n*R*T/V
    Ui = 1.5*n*R*T; dU = 1.5*n*R*dT
    W  = n*R*T*x                             # isothermal work, exact
    Q  = W + dU                              # 1st law, exact
    c  = Q/(m*dT)                            # derived => Q=mc dT exact
    dV = W/P
    eta= W/Q                                 # in (0,1) since dU>0
    Th = LU(400,2000); Tc = Th*(1-eta)
    Vi = LU(.001,2); Vf = Vi*math.exp(x)
    return {"Q_n_mol":n,"Q_R_gas":R,"Q_kB":kB,"Q_T_temp":T,"Q_V_vol":V,
            "Q_m":m,"Q_c_spec":c,"Q_dT":dT,"Q_M_molar":Mm,
            "Q_P_press":P,"Q_U_int":Ui,"Q_dU":dU,"Q_Q_heat":Q,
            "Q_W_th":W,"Q_dV":dV,"Q_eta":eta,"Q_Th":Th,"Q_Tc":Tc,
            "Q_L_lat":Q/m,"Q_dS":Q/T,
            "Q_v_rms":math.sqrt(3*R*T/Mm),"Q_KE_avg":1.5*kB*T,
            "Q_Vi":Vi,"Q_Vf":Vf}

def s12_wave():
    R = CONST_VALS["Q_R_gas"]
    Tt, mu, L = LU(1,500), LU(.001,.5), LU(.2,5)
    A, xw, t  = LU(.001,.2), LU(.01,5), LU(.01,5)
    gam, Mm   = U(1.2,1.67), LU(.002,.2)
    vob, vsr  = U(0,30), U(0,30)
    v  = math.sqrt(Tt/mu)
    f  = v/(2*L); lam = 2*L                  # fundamental, n=1
    om = 2*math.pi*f; kw = 2*math.pi/lam
    f2 = f*U(.7,.95)
    Tg = v*v*Mm/(gam*R)                      # sound-speed consistent
    P  = LU(.1,100); Aa = LU(1e-3,10)
    return {"Q_T_tens":Tt,"Q_mu_lin":mu,"Q_L_str":L,"Q_A_amp":A,
            "Q_x":xw,"Q_t":t,"Q_gamma":gam,"Q_M_molar":Mm,
            "Q_R_gas":R,"Q_n_harm":1.0,"Q_v_wave":v,"Q_f":f,
            "Q_lambda":lam,"Q_omega":om,"Q_k_wave":kw,
            "Q_T_period":1/f,"Q_T_temp":Tg,
            "Q_y_wave":A*math.sin(kw*xw-om*t),
            "Q_f1":f,"Q_f2":f2,"Q_f_beat":abs(f-f2),
            "Q_v_obs":vob,"Q_v_src":vsr,
            "Q_f_obs":f*(v+vob)/(v-vsr),
            "Q_P_pow":P,"Q_A_area":Aa,"Q_I_int":P/Aa}

def s13_matter():
    g, sb = CONST_VALS["Q_g"], CONST_VALS["Q_sigma_SB"]
    F, L0   = LU(1,5000), LU(.1,5)
    eps     = LU(1e-4,5e-3)
    S, r    = LU(.01,.5), LU(1e-4,.05)
    A       = F*r/(2*S)                    # => P=F/A == 2S/r exact
    sig     = F/A
    dL      = eps*L0
    dT      = U(20,400); alpha = eps/dT    # => dL both ways exact
    rho     = LU(800,13600); rho_f = rho*U(.05,.9)
    eta     = LU(1e-3,10); v = LU(.01,5)
    th      = U(0, math.radians(60))
    kth, Tt = LU(.1,400), LU(200,1500)
    return {"Q_F":F,"Q_L0":L0,"Q_eps_str":eps,"Q_A_area":A,
            "Q_sigma_s":sig,"Q_Y_mod":sig/eps,"Q_dL_el":dL,
            "Q_U_el":.5*F*dL,"Q_S_surf":S,"Q_r":r,"Q_g":g,
            "Q_P_press":2*S/r,"Q_theta":th,
            "Q_h_cap":2*S*math.cos(th)/(rho*g*r),
            "Q_rho":rho,"Q_rho_f":rho_f,"Q_eta_visc":eta,"Q_v":v,
            "Q_F_visc":6*math.pi*eta*r*v,
            "Q_v_term":2*r*r*(rho-rho_f)*g/(9*eta),
            "Q_dT":dT,"Q_alpha_x":alpha,"Q_k_therm":kth,
            "Q_P_cond":kth*A*dT/L0,"Q_T_temp":Tt,
            "Q_sigma_SB":sb,"Q_P_rad":sb*A*Tt**4}

SCENARIOS = [("linear",s1_linear),("projectile",s2_projectile),
             ("circular",s3_circular),("incline",s4_incline),
             ("energy",s5_energy),("collision",s6_collision),
             ("rotation",s7_rotation),("orbit",s8_orbit),
             ("shm",s9_shm),("fluid",s10_fluid),
             ("thermo",s11_thermo),("wave",s12_wave),("matter",s13_matter)]

# ════════════════════════════════════════════════════════════════════
# BANNED COMBOS (never trained; emitted as honest chain test set)
#   (scenario, frozenset(revealed_inputs), target)
# ════════════════════════════════════════════════════════════════════
BANS = [
 ("circular", ["Q_f","Q_r"],              "Q_v_tan"),   # L1: f->omega
 ("circular", ["Q_T_period","Q_r"],       "Q_v_tan"),   # L2: T->omega
 ("rotation", ["Q_T_period","Q_r","Q_m"], "Q_KE"),      # L3: T->om->v->KE
 ("shm",      ["Q_k_spr","Q_m","Q_A_amp"],"Q_v_max"),   # k,m->omega chain
 ("wave",     ["Q_lambda","Q_f","Q_mu_lin"],"Q_T_tens"),# v=f*lam -> T=v^2*mu
]

# ════════════════════════════════════════════════════════════════════
# IDENTITY VERIFICATION — the anti-v7 guarantee
# ════════════════════════════════════════════════════════════════════
def verify_world(name, w):
    def ck(label, lhs, rhs, tol=1e-6):
        d = abs(lhs-rhs)/max(abs(rhs), 1e-9)
        assert d < tol, f"[{name}] {label}: {lhs} vs {rhs} (rel {d:.2e})"
    g = CONST_VALS["Q_g"]
    if name=="linear":
        ck("v=u+at", w["Q_v"], w["Q_u"]+w["Q_a"]*w["Q_t"])
        ck("s two-ways", w["Q_s"], .5*(w["Q_u"]+w["Q_v"])*w["Q_t"])
        ck("v2=u2+2as", w["Q_v"]**2, w["Q_u"]**2+2*w["Q_a"]*w["Q_s"])
    if name=="projectile":
        ck("speed", w["Q_v"], math.hypot(w["Q_vx"],w["Q_vy"]))
        ck("range", w["Q_R_range"],
           w["Q_v0"]**2*math.sin(2*w["Q_theta"])/g)
    if name=="circular":
        ck("T*f=1", w["Q_T_period"]*w["Q_f"], 1.0)
        ck("om=2pi f", w["Q_omega"], 2*math.pi*w["Q_f"])
        ck("om^2", w["Q_omega"]**2,
           w["Q_omega0"]**2+2*w["Q_alpha"]*w["Q_phi"])
        ck("atot", w["Q_a_tot"], math.hypot(w["Q_ac"],w["Q_a_tan"]))
    if name=="incline":
        ck("J=dp", w["Q_J_imp"], w["Q_dp"])
        ck("F=ma", w["Q_F"], w["Q_m"]*w["Q_a"])
    if name=="energy":
        ck("P=W/t", w["Q_P_pow"], w["Q_W_work"]/w["Q_t"])
        ck("P=Fv",  w["Q_P_pow"], w["Q_F"]*w["Q_v"])
    if name=="matter":
        ck("Y=sig/eps", w["Q_Y_mod"], w["Q_sigma_s"]/w["Q_eps_str"])
        ck("P=F/A==2S/r", w["Q_P_press"], w["Q_F"]/w["Q_A_area"])
        ck("dL 2 ways", w["Q_dL_el"],
           w["Q_alpha_x"]*w["Q_L0"]*w["Q_dT"])
        ck("v_term", w["Q_v_term"], 2*w["Q_r"]**2*
           (w["Q_rho"]-w["Q_rho_f"])*CONST_VALS["Q_g"]/(9*w["Q_eta_visc"]))
        ck("stefan", w["Q_P_rad"],
           CONST_VALS["Q_sigma_SB"]*w["Q_A_area"]*w["Q_T_temp"]**4)
        ck("conduct", w["Q_P_cond"],
           w["Q_k_therm"]*w["Q_A_area"]*w["Q_dT"]/w["Q_L0"])
    if name=="collision":
        pf = w["Q_m1"]*w["Q_v1f"]+w["Q_m2"]*w["Q_v2f"]
        ck("p conserved", pf, w["Q_p_tot"])
        e  = (w["Q_v2f"]-w["Q_v1f"])/(w["Q_v1i"]-w["Q_v2i"])
        ck("restitution", e, w["Q_e_rest"], 1e-5)
        ck("x_cm", w["Q_x_cm"], (w["Q_m1"]*w["Q_x1"]+w["Q_m2"]*w["Q_x2"])
           /(w["Q_m1"]+w["Q_m2"]), 1e-5)
    if name=="rotation":
        ck("I=mr^2", w["Q_I_rot"], w["Q_m"]*w["Q_r"]**2)
        ck("parallel axis", w["Q_I_rot"],
           w["Q_I_cm"]+w["Q_m"]*w["Q_d_axis"]**2)
        ck("L both ways", w["Q_I_rot"]*w["Q_omega"],
           w["Q_m"]*w["Q_v"]*w["Q_r"])
        ck("tau=rFsin", w["Q_tau"],
           w["Q_r"]*w["Q_F"]*math.sin(w["Q_theta"]))
        ck("L conserved", w["Q_I2"]*w["Q_om2"], w["Q_L_ang"])
    if name=="orbit":
        ck("F=m g_field", w["Q_F_grav"], w["Q_m"]*w["Q_g_field"])
        ck("E=-GMm/2r", w["Q_E_orb"], w["Q_PE_orb"]/2)
    if name=="shm":
        ck("E split", w["Q_E_shm"],
           .5*w["Q_m"]*w["Q_v_shm"]**2+.5*w["Q_k_spr"]*w["Q_x_shm"]**2)
        ck("a=-om^2 x", w["Q_a_shm"], -w["Q_omega"]**2*w["Q_x_shm"], 1e-5)
        ck("T pend", w["Q_T_period"],
           2*math.pi*math.sqrt(w["Q_L_pend"]/g))
        ck("v(A,x)", abs(w["Q_v_shm"]), w["Q_omega"]*math.sqrt(
           max(w["Q_A_amp"]**2-w["Q_x_shm"]**2,0)), 1e-4)
    if name=="fluid":
        ck("P=F/A", w["Q_P_press"], w["Q_F"]/w["Q_A_area"])
        ck("rho=m/V", w["Q_rho"], w["Q_m"]/w["Q_V_vol"])
        ck("torricelli", w["Q_v_fl"], math.sqrt(2*g*w["Q_h"]))
    if name=="thermo":
        ck("1st law", w["Q_dU"], w["Q_Q_heat"]-w["Q_W_th"])
        ck("W=PdV", w["Q_W_th"], w["Q_P_press"]*w["Q_dV"])
        ck("eta carnot", w["Q_eta"], 1-w["Q_Tc"]/w["Q_Th"])
        ck("W iso", w["Q_W_th"], w["Q_n_mol"]*w["Q_R_gas"]*
           w["Q_T_temp"]*math.log(w["Q_Vf"]/w["Q_Vi"]), 1e-5)
    if name=="wave":
        ck("v=f lam", w["Q_v_wave"], w["Q_f"]*w["Q_lambda"])
        ck("v=om/k",  w["Q_v_wave"], w["Q_omega"]/w["Q_k_wave"])
        ck("v sound", w["Q_v_wave"], math.sqrt(
           w["Q_gamma"]*w["Q_R_gas"]*w["Q_T_temp"]/w["Q_M_molar"]))
        ck("f fund", w["Q_f"], w["Q_v_wave"]/(2*w["Q_L_str"]))

def verify_lognorm():
    """mag(F) == mag(m)+mag(a) exactly (up to eps) — structure preserved"""
    m, a = 7.3, 2.1; F = m*a
    d = abs(mag(F) - (mag(m)+mag(a)))
    assert d < 1e-6, f"log-additivity broken: {d}"
    print(f"  log-magnitude additivity: |mag(F)-(mag(m)+mag(a))| = {d:.2e}")

# ════════════════════════════════════════════════════════════════════
# GENERATE
# ════════════════════════════════════════════════════════════════════
def main(n_per):
    total = n_per*len(SCENARIOS)
    mags  = np.zeros((total,NQ), np.float32)
    signs = np.zeros((total,NQ), np.int8)
    exist = np.zeros((total,NQ), bool)
    scen  = np.zeros(total, np.int16)

    print(f"Generating {n_per:,} x {len(SCENARIOS)} scenarios ...")
    row=0
    for si,(name,fn) in enumerate(SCENARIOS):
        w0 = fn(); verify_world(name, w0)          # verify template
        for k in range(n_per):
            w = fn()
            if k < 50: verify_world(name, w)       # spot-verify 50 each
            for q,val in w.items():
                c = QI[q]
                mags[row,c]=mag(val); signs[row,c]=sgn(val)
                exist[row,c]=True
            scen[row]=si; row+=1
        print(f"  {name:11s} ok  ({len(w0)} quantities/world)")
    verify_lognorm()

    np.savez_compressed("scenario_data_v1.npz",
        mags=mags, signs=signs, exist=exist, scen_id=scen)

    # ── honest chain TEST set: exactly the banned reveals ──────────
    NT=1500
    t_mags=[]; t_signs=[]; t_reveal=[]; t_target=[]; t_ban=[]
    sidx={n:i for i,(n,_) in enumerate(SCENARIOS)}
    for bi,(sname,inputs,tgt) in enumerate(BANS):
        fn=dict(SCENARIOS)[sname]
        for _ in range(NT):
            w=fn()
            mrow=np.zeros(NQ,np.float32); srow=np.zeros(NQ,np.int8)
            rrow=np.zeros(NQ,bool)
            for q,val in w.items():
                mrow[QI[q]]=mag(val); srow[QI[q]]=sgn(val)
            for q in inputs: rrow[QI[q]]=True
            for q in CONST_VALS:
                if q in w: rrow[QI[q]]=True
            t_mags.append(mrow); t_signs.append(srow)
            t_reveal.append(rrow); t_target.append(QI[tgt]); t_ban.append(bi)
    np.savez_compressed("scenario_test_v1.npz",
        mags=np.array(t_mags), signs=np.array(t_signs),
        reveal=np.array(t_reveal),
        target=np.array(t_target,np.int32),
        ban_id=np.array(t_ban,np.int16))

    meta = {
      "version":"scenario_v1", "n_samples":int(total),
      "qty_order":QTY, "normalization":
        {"mag":"log10(|x|+1e-12)/10","sign":"+1/-1, 0=absent"},
      "constants":{q:CONST_VALS[q] for q in CONST_VALS},
      "constant_cols":{q:QI[q] for q in CONST_VALS},
      "scenarios":{n:i for i,(n,_) in enumerate(SCENARIOS)},
      "scenario_qtys":{n:sorted(fn().keys()) for n,fn in SCENARIOS},
      "bans":[{"scenario":s,"inputs":i,"target":t} for s,i,t in BANS],
      "train_sampling_rule":
        "reveal = constants + random subset of exist; hide target from "
        "exist; SKIP if (target,reveal\\constants) matches any ban with "
        "reveal ⊆ ban.inputs",
    }
    json.dump(meta, open("scenario_meta_v1.json","w"), indent=1)

    print(f"\nSaved scenario_data_v1.npz  [{total:,} x {NQ}]")
    print(f"Saved scenario_test_v1.npz  [{len(t_target):,} chain tests, "
          f"{len(BANS)} ladders]")
    print(f"Saved scenario_meta_v1.json")

if __name__=="__main__":
    n = int(sys.argv[1]) if len(sys.argv)>1 else 5000
    main(n)
