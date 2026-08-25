"""
PHYSICS GNN TRAINER v9 — operator-conditioned GAT on the full CM graph
=======================================================================
WHAT'S NEW vs v8
  1. FULL-GRAPH forward (305 nodes) — every path always available.
  2. OPERATOR-CONDITIONED MESSAGES: each edge's 16-dim feature vector
     (MUL/ADD/TRIG/POW/DIV/PROP/EXPL + role + dir + NEG/SQ/COEF) mixes
     8 learned basis matrices (R-GCN style) => a value crossing a
     multiply edge is transformed differently than an add/trig edge.
     Reverse edges (all role+dir bits zero) get their own mix.
  3. SCENARIO data: consistent worlds => chains are learnable.
  4. LADDER EXAM built in: 5 banned-combo chain tests (L0..L4) the
     model never saw in training. Passing = routing proven.

ARCHITECTURE
  h0 = LN( SI_embed(x[:7] @ W_base[7,d]) + struct(x[7:]) + type_emb
           + value_or_unknown )
  8 x [ multi-head attention over incoming edges, where per edge:
        m   = sum_b coef_b(e) * (V_b h_src)      <- operator transform
        k,v = W_k m + e_emb ,  W_v m
        q   = W_q h_dst
        alpha = segment_softmax(q.k / sqrt(dh)) over dst's in-edges
        h_dst += W_o( sum alpha v );  + FFN residual ]
  heads: magnitude (Huber) on hidden qty nodes; sign (CE 2-way).

TRAIN-TIME MASKING
  reveal = constants(present) + random 30..80% of existing qtys
  BAN rule enforced: never train (target,reveals) matching a ladder.

RUN
  python gnn_trainer_v9.py            # full training (resumes if ckpt)
  python gnn_trainer_v9.py --smoke    # 2-min CPU sanity run
Checkpoints -> Google Drive (physics_gnn_v9/) when in Colab, else ./ckpt_v9
"""
import os, sys, json, math, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import torch.utils.checkpoint as cp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

SMOKE  = "--smoke" in sys.argv
LOWMEM = "--lowmem" in sys.argv

# ── config ──────────────────────────────────────────────────────────
class C:
    d        = 128
    layers   = 3 if SMOKE else 8
    heads    = 4
    n_bases  = 8
    batch    = 64 if SMOKE else (64 if LOWMEM else 128)
    accum    = 1 if SMOKE else 2
    epochs   = 1 if SMOKE else int(os.environ.get("PHYSICS_GNN_EPOCHS", "60"))
    lr       = 3e-4
    wd       = 0.01
    warmup   = 50 if SMOKE else 1000
    clip     = 1.0
    val_frac = 0.05
    sign_w   = 0.2
    ladder_every = 1 if SMOKE else 2
    seed     = int(os.environ.get("PHYSICS_GNN_SEED", "7"))

torch.manual_seed(C.seed); np.random.seed(C.seed)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEV == "cuda"

# ── storage: Drive on Colab, local otherwise ────────────────────────
def setup_dirs():
    override = os.environ.get("PHYSICS_GNN_CKDIR")
    if override:
        os.makedirs(override, exist_ok=True)
        return override
    if os.path.exists("/content"):
        try:
            if not os.path.exists("/content/drive/MyDrive"):
                from google.colab import drive
                drive.mount("/content/drive")
            ck = "/content/drive/MyDrive/physics_gnn_v9"
            os.makedirs(ck, exist_ok=True)
            return ck
        except Exception as e:
            print("Drive unavailable:", e)
    ck = "./ckpt_v9"; os.makedirs(ck, exist_ok=True)
    return ck

CKDIR = setup_dirs()
print(f"device={DEV}  amp={AMP}  checkpoints -> {CKDIR}")

# ── load graph ──────────────────────────────────────────────────────
G = json.load(open("cm_fullgraph_v1.json"))
X_np   = np.array(G["X"], np.float32)                  # [N,32]
EI_np  = np.array(G["edge_index"], np.int64).T         # [2,E]
EF_np  = np.array(G["E"], np.float32)                  # [E,16]
NODES  = len(X_np); EDGES = EI_np.shape[1]
ntype  = {"law":0, "quantity":1, "equation":2}
TYPE_np= np.array([ntype[m["node_type"]] for m in G["node_metadata"]],
                  np.int64)
QTY_IDS  = [m["id"] for m in G["node_metadata"]
            if m["node_type"]=="quantity"]
QNODE_np = np.array([m["index"] for m in G["node_metadata"]
                     if m["node_type"]=="quantity"], np.int64)  # [Q]

# ── load data ───────────────────────────────────────────────────────
meta = json.load(open("scenario_meta_v1.json"))
assert meta["qty_order"] == QTY_IDS, "data/graph quantity order mismatch"
D = np.load("scenario_data_v1.npz")
MAGS, SIGNS = D["mags"], D["signs"]
EXIST       = D["exist"]
NQ = MAGS.shape[1]
CONST_COLS  = np.zeros(NQ, bool)
for q,c in meta["constant_cols"].items(): CONST_COLS[c] = True

if SMOKE:
    keep = np.random.default_rng(0).choice(len(MAGS), 4000, replace=False)
    MAGS, SIGNS, EXIST = MAGS[keep], SIGNS[keep], EXIST[keep]

n_val  = int(len(MAGS)*C.val_frac)
perm   = np.random.default_rng(1).permutation(len(MAGS))
VA, TR = perm[:n_val], perm[n_val:]
print(f"graph: {NODES} nodes / {EDGES} edges | data: {len(TR):,} train "
      f"/ {len(VA):,} val | quantities: {NQ}")

# bans -> (input col array, target col)
QCOL = {q:i for i,q in enumerate(QTY_IDS)}
BANS = [ (np.array([QCOL[q] for q in b["inputs"]]), QCOL[b["target"]])
         for b in meta["bans"] ]

T = np.load("scenario_test_v1.npz")
T_MAGS, T_SIGNS = T["mags"], T["signs"]
T_REVEAL, T_TGT, T_BID = T["reveal"], T["target"], T["ban_id"]
if SMOKE:
    T_MAGS,T_SIGNS = T_MAGS[::10], T_SIGNS[::10]
    T_REVEAL,T_TGT,T_BID = T_REVEAL[::10], T_TGT[::10], T_BID[::10]

# ── static graph tensors on device ──────────────────────────────────
Xg    = torch.tensor(X_np, device=DEV)
EIg   = torch.tensor(EI_np, device=DEV)
EFg   = torch.tensor(EF_np, device=DEV)
TYPEg = torch.tensor(TYPE_np, device=DEV)
QNODE = torch.tensor(QNODE_np, device=DEV)
CONSTt= torch.tensor(CONST_COLS, device=DEV)
BANS_t= [(torch.tensor(i, device=DEV), t) for i,t in BANS]

def batch_graph(B):
    """edge_index for B stacked copies of the graph."""
    off = (torch.arange(B, device=DEV)*NODES).repeat_interleave(EDGES)
    ei  = EIg.repeat(1, B) + off
    return ei
EI_FULL = batch_graph(C.batch)          # cached for full batches
EF_FULL = EFg.repeat(C.batch, 1)

# ════════════════════════════════════════════════════════════════════
# MODEL
# ════════════════════════════════════════════════════════════════════
def segment_softmax(score, dst, n):
    """softmax of score [E,H] grouped by dst node."""
    H = score.shape[1]
    idx = dst.unsqueeze(-1).expand(-1, H)
    mx  = torch.full((n, H), -1e30, device=score.device, dtype=score.dtype)
    mx  = mx.scatter_reduce(0, idx, score, reduce="amax", include_self=True)
    ex  = torch.exp(score - mx[dst])
    sm  = torch.zeros((n, H), device=score.device, dtype=score.dtype)
    sm  = sm.scatter_add(0, idx, ex)
    return ex / (sm[dst] + 1e-16)

class OpGATLayer(nn.Module):
    def __init__(s, d, heads, nb):
        super().__init__()
        s.d, s.h, s.dh, s.nb = d, heads, d//heads, nb
        s.coef  = nn.Linear(16, nb)
        s.V     = nn.Parameter(torch.randn(nb, d, d) * (1/math.sqrt(d)))
        s.e_emb = nn.Linear(16, d)
        s.Wq, s.Wk, s.Wv = (nn.Linear(d, d) for _ in range(3))
        s.Wo    = nn.Linear(d, d)
        s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.ffn   = nn.Sequential(nn.Linear(d, 2*d), nn.GELU(),
                                nn.Linear(2*d, d))
        s.drop  = nn.Dropout(0.1)

    def forward(s, h, ei, ef):
        src, dst = ei
        n = h.shape[0]
        hs = h[src]                                   # [E,d]
        c  = s.coef(ef)                               # [E,nb]
        W  = s.V.transpose(1, 2).permute(1, 0, 2)     # [d,nb,d]
        hsV= hs @ W.reshape(s.d, s.nb*s.d)            # ONE matmul [E,nb*d]
        m  = (c.unsqueeze(-1) * hsV.view(-1, s.nb, s.d)).sum(1)
        e  = s.e_emb(ef)
        q  = s.Wq(h)[dst].view(-1, s.h, s.dh)
        k  = s.Wk(m + e).view(-1, s.h, s.dh)
        v  = s.Wv(m).view(-1, s.h, s.dh)
        a  = segment_softmax((q*k).sum(-1)/math.sqrt(s.dh), dst, n)
        out= torch.zeros(n, s.h, s.dh, device=h.device, dtype=h.dtype)
        out= out.scatter_add(0, dst.view(-1,1,1).expand(-1,s.h,s.dh),
                             a.unsqueeze(-1)*v)
        h  = s.ln1(h + s.drop(s.Wo(out.reshape(n, s.d))))
        h  = s.ln2(h + s.drop(s.ffn(h)))
        return h

class PhysicsGNN(nn.Module):
    def __init__(s, d=C.d, layers=C.layers, heads=C.heads, nb=C.n_bases):
        super().__init__()
        s.W_base   = nn.Parameter(torch.randn(7, d)*0.1)   # SI embeddings
        s.struct   = nn.Linear(25, d)
        s.type_emb = nn.Embedding(3, d)
        s.val_in   = nn.Linear(2, d)
        s.unk      = nn.Parameter(torch.randn(d)*0.1)
        s.ln0      = nn.LayerNorm(d)
        s.layers   = nn.ModuleList(OpGATLayer(d,heads,nb)
                                   for _ in range(layers))
        s.mag_head = nn.Sequential(nn.Linear(d,d), nn.GELU(),
                                   nn.Linear(d,1))
        s.sgn_head = nn.Sequential(nn.Linear(d,d), nn.GELU(),
                                   nn.Linear(d,2))

    def forward(s, mag_n, sgn_n, known_n, B, ei):
        # static structural part [N,d] -> expand to batch
        st = (Xg[:, :7] @ s.W_base) + s.struct(Xg[:, 7:]) \
             + s.type_emb(TYPEg)                       # [N,d]
        st = st.unsqueeze(0).expand(B, -1, -1)         # [B,N,d]
        val = s.val_in(torch.stack([mag_n, sgn_n], -1))# [B,N,d]
        kn  = known_n.unsqueeze(-1)
        h   = s.ln0(st + kn*val + (1-kn)*s.unk)        # [B,N,d]
        h   = h.reshape(B*NODES, -1)
        ef  = EF_FULL if B == C.batch else EFg.repeat(B, 1)
        for lyr in s.layers:
            if LOWMEM and s.training:
                h = cp.checkpoint(lyr, h, ei, ef, use_reentrant=False)
            else:
                h = lyr(h, ei, ef)
        return s.mag_head(h).squeeze(-1), s.sgn_head(h)

# ── batch assembly ──────────────────────────────────────────────────
def make_batch(rows, reveal_q, train=True):
    """rows: np idx  reveal_q: torch bool [B,NQ] -> node tensors+targets"""
    B = len(rows)
    mag_q  = torch.tensor(MAGS[rows],  device=DEV)          # [B,NQ]
    sgn_q  = torch.tensor(SIGNS[rows], device=DEV, dtype=torch.float32)
    ex_q   = torch.tensor(EXIST[rows], device=DEV)
    hid_q  = ex_q & ~reveal_q
    mag_n  = torch.zeros(B, NODES, device=DEV)
    sgn_n  = torch.zeros(B, NODES, device=DEV)
    kn_n   = torch.zeros(B, NODES, device=DEV)
    qn     = QNODE.unsqueeze(0).expand(B, -1)
    mag_n.scatter_(1, qn, mag_q * reveal_q)
    sgn_n.scatter_(1, qn, sgn_q * reveal_q)
    kn_n.scatter_(1, qn, reveal_q.float())
    return mag_n, sgn_n, kn_n, mag_q, sgn_q, hid_q

def sample_reveal(rows):
    B = len(rows)
    ex   = torch.tensor(EXIST[rows], device=DEV)
    frac = torch.rand(B, 1, device=DEV)*0.5 + 0.3
    rev  = ex & ((torch.rand(B, NQ, device=DEV) < frac) | CONSTt)
    hid  = ex & ~rev
    # guarantee >=1 hidden
    none = hid.sum(1) == 0
    if none.any():
        cand = ex[none] & ~CONSTt
        pick = torch.multinomial(cand.float()+1e-9, 1).squeeze(1)
        rev[none.nonzero(as_tuple=True)[0], pick] = False
    # BAN enforcement: reveal an extra non-ban qty on violating rows
    for icols, tcol in BANS_t:
        ban_in = torch.zeros(NQ, dtype=torch.bool, device=DEV)
        ban_in[icols] = True
        hid = ex & ~rev
        viol = hid[:, tcol] & ((rev & ~CONSTt & ~ban_in).sum(1) == 0)
        if viol.any():
            cand = ex[viol] & ~rev[viol] & ~ban_in
            cand[:, tcol] = False
            ok = cand.sum(1) > 0
            vidx = viol.nonzero(as_tuple=True)[0][ok]
            pick = torch.multinomial(cand[ok].float()+1e-9, 1).squeeze(1)
            rev[vidx, pick] = True
    return rev

# ════════════════════════════════════════════════════════════════════
# LOSS / EVAL
# ════════════════════════════════════════════════════════════════════
def compute_loss(pm, ps, mag_q, sgn_q, hid_q, B):
    pm_q = pm.view(B, NODES).gather(1, QNODE.unsqueeze(0).expand(B,-1))
    ps_q = ps.view(B, NODES, 2).gather(
        1, QNODE.view(1,-1,1).expand(B,-1,2))
    hm   = hid_q
    l_mag = F.huber_loss(pm_q[hm], mag_q[hm]) if hm.any() else pm.sum()*0
    hs   = hid_q & (sgn_q != 0)
    if hs.any():
        tgt = (sgn_q[hs] > 0).long()
        l_sgn = F.cross_entropy(ps_q[hs], tgt)
        acc   = (ps_q[hs].argmax(-1) == tgt).float().mean().item()
    else:
        l_sgn, acc = pm.sum()*0, 1.0
    mae = (pm_q[hm]-mag_q[hm]).abs().mean().item() if hm.any() else 0.
    return l_mag + C.sign_w*l_sgn, mae, acc

@torch.no_grad()
def ladder_exam(model):
    model.eval(); res = {}
    for bi in range(len(BANS)):
        sel = np.where(T_BID == bi)[0]
        preds, tgts = [], []
        for i0 in range(0, len(sel), C.batch):
            rows = sel[i0:i0+C.batch]; B = len(rows)
            mag_q = torch.tensor(T_MAGS[rows], device=DEV)
            sgn_q = torch.tensor(T_SIGNS[rows], device=DEV,
                                 dtype=torch.float32)
            rev   = torch.tensor(T_REVEAL[rows], device=DEV)
            mag_n = torch.zeros(B, NODES, device=DEV)
            sgn_n = torch.zeros(B, NODES, device=DEV)
            kn_n  = torch.zeros(B, NODES, device=DEV)
            qn    = QNODE.unsqueeze(0).expand(B, -1)
            mag_n.scatter_(1, qn, mag_q*rev)
            sgn_n.scatter_(1, qn, sgn_q*rev)
            kn_n.scatter_(1, qn, rev.float())
            ei = EI_FULL if B == C.batch else batch_graph(B)
            with torch.autocast(DEV, enabled=AMP):
                pm, _ = model(mag_n, sgn_n, kn_n, B, ei)
            pm_q = pm.view(B, NODES).gather(1, qn)
            tcol = torch.tensor(T_TGT[rows], device=DEV).unsqueeze(1)
            preds.append(pm_q.gather(1, tcol).squeeze(1).float().cpu())
            tgts.append(mag_q.gather(1, tcol).squeeze(1).cpu())
        p = torch.cat(preds).numpy(); t = torch.cat(tgts).numpy()
        ss = ((t-t.mean())**2).sum()
        res[f"L{bi}"] = float(1 - ((p-t)**2).sum()/max(ss,1e-12))
    model.train(); return res

@torch.no_grad()
def validate(model):
    model.eval(); maes, n = 0., 0
    for i0 in range(0, len(VA), C.batch):
        rows = VA[i0:i0+C.batch]
        if len(rows) < C.batch: break
        rev = sample_reveal(rows)
        mn, sn, kn, mq, sq, hq = make_batch(rows, rev)
        with torch.autocast(DEV, enabled=AMP):
            pm, ps = model(mn, sn, kn, len(rows), EI_FULL)
            _, mae, _ = compute_loss(pm, ps, mq, sq, hq, len(rows))
        maes += mae; n += 1
    model.train(); return maes/max(n,1)

# ════════════════════════════════════════════════════════════════════
# TRAIN
# ════════════════════════════════════════════════════════════════════
def main():
    model = PhysicsGNN().to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"model: {nparam/1e6:.2f}M params | layers={C.layers} "
          f"d={C.d} heads={C.heads} bases={C.n_bases}")
    opt = torch.optim.AdamW(model.parameters(), lr=C.lr, weight_decay=C.wd)
    micro_ep = len(TR)//C.batch
    steps_ep = micro_ep//C.accum
    total    = steps_ep*C.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(
        (s+1)/C.warmup, 0.5*(1+math.cos(math.pi*min(s/total,1.)))))
    scaler = torch.amp.GradScaler(enabled=AMP)

    ep0, step, best, hist = 0, 0, 1e9, []
    ck_last = os.path.join(CKDIR, "v9_latest.pt")
    ck_best = os.path.join(CKDIR, "v9_best.pt")
    for ck in (ck_last, ck_best):
        if os.path.exists(ck):
            print("resuming from", ck)
            st = torch.load(ck, map_location=DEV, weights_only=False)
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
            ep0, step, best, hist = st["epoch"], st["step"], st["best"], st["hist"]
            break

    rngp = np.random.default_rng(100+ep0)
    for ep in range(ep0, C.epochs):
        order = rngp.permutation(TR)
        t0, tl, tm, ta, nb = time.time(), 0., 0., 0., 0
        opt.zero_grad(set_to_none=True)
        for mi in range(micro_ep):
            rows = order[mi*C.batch:(mi+1)*C.batch]
            rev  = sample_reveal(rows)
            mn, sn, kn, mq, sq, hq = make_batch(rows, rev)
            with torch.autocast(DEV, enabled=AMP):
                pm, ps = model(mn, sn, kn, C.batch, EI_FULL)
                loss, mae, acc = compute_loss(pm, ps, mq, sq, hq, C.batch)
            scaler.scale(loss/C.accum).backward()
            tl+=loss.item(); tm+=mae; ta+=acc; nb+=1
            if (mi+1) % C.accum == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), C.clip)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True); sched.step(); step+=1
            if nb == 1 and DEV=="cuda":
                print(f"  peak GPU mem: "
                      f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB",
                      flush=True)
            if SMOKE and nb >= 40: break
            if nb % 400 == 0:
                print(f"  ep{ep} {nb}/{micro_ep} loss={tl/nb:.4f} "
                      f"mae={tm/nb:.4f} sgn={ta/nb:.3f} "
                      f"({(time.time()-t0)/nb:.2f}s/it)", flush=True)
        vmae = validate(model)
        row = dict(ep=ep, loss=tl/nb, mae=tm/nb, sgn=ta/nb, val_mae=vmae)
        if ep % C.ladder_every == 0 or ep == C.epochs-1:
            lad = ladder_exam(model); row.update(lad)
            print(f"EPOCH {ep}  loss={row['loss']:.4f} val_mae={vmae:.4f} "
                  f"| LADDER " + " ".join(f"{k}={v:+.3f}"
                  for k,v in lad.items()), flush=True)
        else:
            print(f"EPOCH {ep}  loss={row['loss']:.4f} "
                  f"val_mae={vmae:.4f}", flush=True)
        hist.append(row)
        st = dict(model=model.state_dict(), opt=opt.state_dict(),
                  sched=sched.state_dict(), scaler=scaler.state_dict(),
                  epoch=ep+1, step=step, best=best, hist=hist)
        torch.save(st, ck_last)
        if vmae < best:
            best = vmae; st["best"] = best; torch.save(st, ck_best)
        json.dump(hist, open(os.path.join(CKDIR,"v9_history.json"),"w"))
    print("done. best val_mae:", best)

if __name__ == "__main__":
    main()
