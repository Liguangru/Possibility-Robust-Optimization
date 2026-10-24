# -*- coding: utf-8 -*-
"""
SPE (Sparse Possibility Evaluation, de Jong 2026 PMB 71 095016) + CVaR-constrained robust optimization - methodology demo
Framework: minimize upper-CVaR of OAR mean dose, subject to two-sided CVaR constraints on CTV (TXT scheme)
Acceleration: SPE - precompute dose on an error grid; map each fraction's error to nearest grid point; course dose remains linear in spot weights
Solver: Rockafellar-Uryasev linearization => both course-level and voxel-level CVaR expanded into auxiliary variables, overall an LP (solved with HiGHS)
Note: Target constraints carry elastic slack (large penalty) to guarantee feasibility
Note: 2D square phantom + toy Bragg curve; methodology demo only, not clinical dose.
"""
import time
import numpy as np
import matplotlib
from matplotlib import font_manager
for _f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]:
    font_manager.fontManager.addfont(_f)
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy import sparse

rng = np.random.default_rng(7)
OUT = "/mnt/agents/output"

# ============ 1. Phantom and structures ============
N = 28
axc = np.arange(N) + 0.5
X, Y = np.meshgrid(axc, axc)
ctv = (X - 14.5)**2 + (Y - 13.5)**2 <= 8.0**2
oar = ((X - 25.5)**2 + (Y - 13.5)**2 <= 4.0**2) & ~ctv
idx_ctv = np.flatnonzero(ctv.ravel())
idx_oar = np.flatnonzero(oar.ravel())
idx_struct = np.flatnonzero((ctv | oar).ravel())
P = np.column_stack([X.ravel()[idx_struct], Y.ravel()[idx_struct]])
V = P.shape[0]
ctv_pos = np.searchsorted(idx_struct, idx_ctv)
oar_pos = np.searchsorted(idx_struct, idx_oar)
n_ctv, n_oar = len(ctv_pos), len(oar_pos)

# ============ 2. Toy pencil-beam model ============
ISO = np.array([14.5, 13.5])
ANGLES = np.deg2rad([90.0, 210.0, 330.0])
SIG_LAT = 3.0                                            # lateral spot sigma (mm)
R_LAYERS = np.array([15.5, 18.0, 20.5, 23.0, 25.5, 28.0, 30.5])  # 7 energy layers
O_LAT = np.arange(-12.0, 12.1, 3.0)                      # 9 lateral positions
Z0 = 20.0

def bragg(z, R):
    z = np.asarray(z, float)
    B = np.zeros_like(z)
    m = z <= R
    B[m] = 0.25 + 2.75 * np.exp(-((R - z[m]) / 2.2)**2)
    d = z[~m] - R
    B[~m] = 2.75 * np.exp(-(d / 0.9)**2) + 0.25 * np.exp(-d / 1.5)
    B[z > R + 4.0] = 0.0
    return B

spots = []
for th in ANGLES:
    u = np.array([np.cos(th), np.sin(th)])
    v = np.array([-np.sin(th), np.cos(th)])
    for o in O_LAT:
        for R in R_LAYERS:
            spots.append((u, v, o, R))
n_spots = len(spots)                                     # 3 x 9 x 7 = 189

def struct_DI(shift, rho):
    r = P + np.asarray(shift, float)
    D = np.empty((V, n_spots))
    for j, (u, v, o, R) in enumerate(spots):
        rel = r - ISO
        z = (rel @ u + Z0) * (1.0 + rho / 100.0)
        lat = rel @ v - o
        D[:, j] = bragg(z, R) * np.exp(-lat**2 / (2 * SIG_LAT**2))
    return D

# ============ 3. SPE error grid ============
s_sys, s_rand, s_rng = 0.92, 1.0, 1.5          # mm, mm, %
s_comb = np.hypot(s_sys, s_rand)
Emax = 3 * s_comb
g1d = np.linspace(-Emax, Emax, 7)              # order 7
gx, gy = np.meshgrid(g1d, g1d)
keep = (gx**2 + gy**2) <= Emax**2 + 1e-9
setup_grid = np.column_stack([gx[keep], gy[keep]])       # 29 points
range_grid = np.arange(-4.5, 4.51, 1.5)                  # 7 points
G = [(sx, sy, rr) for (sx, sy) in setup_grid for rr in range_grid]
n_grid = len(G)                                          # 203

t0 = time.time()
Dg = np.stack([struct_DI((sx, sy), rr) for (sx, sy, rr) in G])
t_pre = time.time() - t0
nom = next(i for i, (sx, sy, rr) in enumerate(G)
           if abs(sx) < 1e-12 and abs(sy) < 1e-12 and abs(rr) < 1e-12)
Dg_flat = Dg.reshape(n_grid, -1)

# ============ 4. Course scenario sampling + nearest-neighbor mapping ============
K, F = 100, 35

def sample_courses(Kn):
    return (rng.normal(0, s_sys, (Kn, 2)),
            rng.normal(0, s_rng, Kn),
            rng.normal(0, s_rand, (Kn, F, 2)))

def snap_counts(Sys, Rho, Rand):
    Kn = Sys.shape[0]
    C = np.zeros((Kn, n_grid))
    for k in range(Kn):
        tot = Sys[k][None, :] + Rand[k]
        si = ((tot[:, None, :] - setup_grid[None])**2).sum(-1).argmin(1)
        ri = np.argmin(np.abs(range_grid - Rho[k]))
        C[k] = np.bincount(si * len(range_grid) + ri, minlength=n_grid)
    return C

Sys, Rho, Rand = sample_courses(K)
C = snap_counts(Sys, Rho, Rand)
Tc = ((C / F) @ Dg_flat).reshape(K, V, n_spots)          # per-course structure DI (linear in x)

# ============ 5. RU linearization LP (elastic constraints) ============
D_RX, DELTA, ALPHA = 60.0, 0.05, 0.10
LOW_FRAC, HOT_FRAC = 0.20, 0.05
ctv_sub = ctv_pos[np.sort(rng.choice(n_ctv, size=60, replace=False))]  # voxel subsampling
nc = len(ctv_sub)
LAM = 200.0                                              # elastic penalty

def solve_lp(Tc_in, delta):
    Kc, _, n = Tc_in.shape
    nl = max(1, int(round(LOW_FRAC * nc))); nh = max(1, int(round(HOT_FRAC * nc)))
    Tct = Tc_in[:, ctv_sub, :]; A_oar = Tc_in[:, oar_pos, :].mean(1)
    i_x = 0
    i_eO = n;             i_s = i_eO + 1
    i_eL = i_s + Kc;      i_w = i_eL + 1
    i_z = i_w + Kc;       i_u = i_z + Kc
    i_eH = i_u + Kc*nc;   i_q = i_eH + 1
    i_t = i_q + Kc;       i_p = i_t + Kc
    i_e1 = i_p + Kc*nc;   i_e2 = i_e1 + 1
    nvar = i_e2 + 1
    inv_aK = 1.0 / (ALPHA * Kc)
    c = np.zeros(nvar); c[i_eO] = 1.0; c[i_s:i_s+Kc] = inv_aK
    c[i_e1] = LAM; c[i_e2] = LAM

    rows, cols, vals, rhs = [], [], [], []
    r = 0
    def push(rr_, cc, vv):
        rows.append(np.atleast_1d(rr_)); cols.append(np.atleast_1d(cc)); vals.append(np.atleast_1d(vv))
    def adv(nrow, b):
        nonlocal r
        rhs.extend([b] * nrow); r += nrow

    for k in range(Kc):                                   # s_k >= om_k - eO
        push(np.full(n+2, r), np.r_[np.arange(n), i_eO, i_s+k], np.r_[A_oar[k], -1., -1.]); adv(1, 0.)
    push(np.full(Kc+2, r), np.r_[i_eL, i_w+np.arange(Kc), i_e1],
         np.r_[-1., np.full(Kc, inv_aK), -1.]); adv(1, -D_RX*(1-delta))   # low-tail constraint + slack e1
    for k in range(Kc):
        push(np.full(nc+3, r), np.r_[i_w+k, i_z+k, i_eL, i_u+k*nc+np.arange(nc)],
             np.r_[-1., -1., 1., np.full(nc, 1./nl)]); adv(1, 0.)
    push(np.full(Kc+2, r), np.r_[i_eH, i_q+np.arange(Kc), i_e2],
         np.r_[1., np.full(Kc, inv_aK), -1.]); adv(1, D_RX*(1+delta))     # hot-tail constraint + slack e2
    for k in range(Kc):
        push(np.full(nc+3, r), np.r_[i_t+k, i_eH, i_q+k, i_p+k*nc+np.arange(nc)],
             np.r_[1., -1., -1., np.full(nc, 1./nh)]); adv(1, 0.)
    rr_ = np.arange(Kc*nc); Rb = np.repeat(np.arange(Kc*nc), n)
    push(Rb + r, np.tile(np.arange(n), Kc*nc), -Tct.reshape(-1))
    push(rr_ + r, i_z + np.repeat(np.arange(Kc), nc), np.ones(Kc*nc))
    push(rr_ + r, i_u + rr_, -np.ones(Kc*nc)); adv(Kc*nc, 0.)
    push(Rb + r, np.tile(np.arange(n), Kc*nc), Tct.reshape(-1))
    push(rr_ + r, i_t + np.repeat(np.arange(Kc), nc), -np.ones(Kc*nc))
    push(rr_ + r, i_p + rr_, -np.ones(Kc*nc)); adv(Kc*nc, 0.)

    A_ub = sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(r, nvar)).tocsr()
    b_ub = np.array(rhs, float)
    bounds = ([(0, None)] * n + [(None, None)] + [(0, None)] * Kc + [(None, None)]
              + [(0, None)] * Kc + [(None, None)] * Kc + [(0, None)] * Kc*nc
              + [(None, None)] + [(0, None)] * Kc + [(None, None)] * Kc + [(0, None)] * Kc*nc
              + [(0, None), (0, None)])
    t0 = time.time()
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return res, nvar, time.time() - t0, (i_e1, i_e2)

t0 = time.time()
res_r, nvar_r, t_lp, (i_e1, i_e2) = solve_lp(Tc, DELTA)
x_rob = res_r.x[:n_spots]
res_d, _, t_det, _ = solve_lp(Dg[nom:nom+1], DELTA)      # deterministic reference (K=1, correct approach)
x_det = res_d.x[:n_spots]
print(f"Robust LP:  {t_lp:5.1f} s  status={res_r.message.strip()[:40]}")
print(f"  elastic violation: low-tail e1={res_r.x[i_e1]:.3f} Gy, hot-tail e2={res_r.x[i_e2]:.3f} Gy")
print(f"Deterministic LP: {t_det:5.1f} s")

# ============ 6. Independent validation: SPE vs exact per-fraction computation ============
Ke = 100
SysE, RhoE, RandE = sample_courses(Ke)
nl = max(1, int(round(LOW_FRAC * n_ctv))); nh = max(1, int(round(HOT_FRAC * n_ctv)))

def eval_metrics(x, mode):
    om = np.empty(Ke); cl = np.empty(Ke); ch = np.empty(Ke)
    if mode == "spe":
        Ce = snap_counts(SysE, RhoE, RandE)
        d = ((Ce / F) @ Dg_flat).reshape(Ke, V, n_spots) @ x
    else:
        d = np.empty((Ke, V))
        for k in range(Ke):
            acc = np.zeros(V)
            for f in range(F):
                acc += struct_DI(SysE[k] + RandE[k, f], RhoE[k]) @ x
            d[k] = acc / F
    for k in range(Ke):
        srt = np.sort(d[k, ctv_pos])
        om[k] = d[k, oar_pos].mean(); cl[k] = srt[:nl].mean(); ch[k] = srt[-nh:].mean()
    return om, cl, ch

t0 = time.time(); om_spe, cl_spe, ch_spe = eval_metrics(x_rob, "spe");   t_spe = time.time() - t0
t0 = time.time(); om_ex, cl_ex, ch_ex = eval_metrics(x_rob, "exact");    t_ex = time.time() - t0
om_det, cl_det, ch_det = eval_metrics(x_det, "exact")

pct = np.arange(1, 100)
mpe_cl = np.mean(np.abs(np.percentile(cl_spe, pct) - np.percentile(cl_ex, pct)))
mpe_om = np.mean(np.abs(np.percentile(om_spe, pct) - np.percentile(om_ex, pct)))
cvarl = lambda v: v[v <= np.quantile(v, ALPHA)].mean()
cvaru = lambda v: v[v >= np.quantile(v, 1 - ALPHA)].mean()

print(f"\n=== Runtime ===")
print(f"SPE grid precompute ({n_grid} doses):  {t_pre:5.1f} s")
print(f"SPE evaluation {Ke} courses:               {t_spe:5.2f} s")
print(f"Exact evaluation {Ke} courses ({Ke*F} fractions):  {t_ex:5.1f} s   speedup ~{t_ex/t_spe:.0f}x")
print(f"\n=== SPE accuracy (vs exact, MPE) ===")
print(f"CTV low-tail MPE: {mpe_cl:.3f} Gy | OAR mean MPE: {mpe_om:.3f} Gy")
print(f"\n=== Plan comparison (independent exact evaluation, {Ke} courses) ===")
print(f"{'Metric':26s} {'Deterministic':>10s} {'SPE-Robust':>12s}")
for lab, a, b in [("CTV low-tail CVaR_lower (>=57)", cvarl(cl_det), cvarl(cl_ex)),
                  ("CTV low-tail P10", np.percentile(cl_det, 10), np.percentile(cl_ex, 10)),
                  ("CTV low-tail worst course", cl_det.min(), cl_ex.min()),
                  ("CTV hot-tail CVaR_upper (<=63)", cvaru(ch_det), cvaru(ch_ex)),
                  ("OAR mean CVaR_upper (objective)", cvaru(om_det), cvaru(om_ex)),
                  ("OAR mean P90", np.percentile(om_det, 90), np.percentile(om_ex, 90))]:
    print(f"{lab:26s} {a:10.2f} {b:12.2f}")

# ============ 7. Figures ============
Pa = np.column_stack([X.ravel(), Y.ravel()])
def full_dose(x):
    D = np.zeros(N * N)
    for j, (u, v, o, R) in enumerate(spots):
        rel = Pa - ISO
        z = (rel @ u + Z0); lat = rel @ v - o
        D += x[j] * bragg(z, R) * np.exp(-lat**2 / (2 * SIG_LAT**2))
    return D.reshape(N, N)

fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
ax = axs[0]
ax.contour(X, Y, ctv, colors="tab:red", linewidths=2)
ax.contour(X, Y, oar, colors="tab:blue", linewidths=2)
ax.plot(setup_grid[:, 0] + ISO[0], setup_grid[:, 1] + ISO[1], "k.", ms=4)
for th in ANGLES:
    u = np.array([np.cos(th), np.sin(th)])
    ax.annotate("", xy=ISO + 6 * u, xytext=ISO + 16 * u,
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
ax.plot(*ISO, "k+"); ax.set_title("Geometry (red=CTV, blue=OAR, black dots=SPE grid)"); ax.set_aspect("equal")
zz = np.linspace(0, 34, 400)
axs[1].plot(zz, bragg(zz, 24.0)); axs[1].set_title("Toy Bragg curve (R=24 mm)")
axs[1].set_xlabel("Depth (mm)")
im = axs[2].imshow(full_dose(x_rob), cmap="jet", origin="lower")
axs[2].contour(ctv, colors="white", linewidths=1.2)
axs[2].contour(oar, colors="cyan", linewidths=1.2)
axs[2].set_title("SPE robust plan nominal dose"); plt.colorbar(im, ax=axs[2], label="Gy")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_geometry_and_dose_model.png", dpi=150); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
axs[0].plot(np.percentile(cl_ex, pct), pct, "k-", lw=2, label="Exact (3500 fractions)")
axs[0].plot(np.percentile(cl_spe, pct), pct, "r--", lw=2, label=f"SPE ({n_grid} grid doses)")
axs[0].axvline(D_RX * 0.95, color="gray", ls=":", label="Constraint lower bound 57 Gy")
axs[0].set_xlabel("CTV low-tail dose (Gy)"); axs[0].set_ylabel("Percentile")
axs[0].set_title(f"CTV coverage | MPE={mpe_cl:.2f} Gy"); axs[0].legend()
axs[1].plot(np.percentile(om_ex, pct), pct, "k-", lw=2, label="Exact")
axs[1].plot(np.percentile(om_spe, pct), pct, "r--", lw=2, label="SPE")
axs[1].set_xlabel("OAR mean dose (Gy)"); axs[1].set_ylabel("Percentile")
axs[1].set_title(f"OAR | MPE={mpe_om:.2f} Gy"); axs[1].legend()
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_spe_accuracy_validation.png", dpi=150); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
for v, lab, c_ in [(cl_det, "Deterministic plan", "tab:blue"), (cl_ex, "SPE robust plan", "tab:red")]:
    axs[0].plot(np.sort(v), np.linspace(0, 1, Ke), lw=2, color=c_, label=lab)
axs[0].axvline(D_RX * 0.95, color="gray", ls=":")
axs[0].set_xlabel("CTV low-tail dose (Gy)"); axs[0].set_ylabel("Cumulative probability")
axs[0].set_title("Target: worst tail lifted"); axs[0].legend()
for v, lab, c_ in [(om_det, "Deterministic plan", "tab:blue"), (om_ex, "SPE robust plan", "tab:red")]:
    axs[1].plot(np.sort(v), np.linspace(0, 1, Ke), lw=2, color=c_, label=lab)
axs[1].set_xlabel("OAR mean dose (Gy)"); axs[1].set_ylabel("Cumulative probability")
axs[1].set_title("OAR: cost of robustness"); axs[1].legend()
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_robust_optimization_results.png", dpi=150); plt.close(fig)
print("\nFigures saved.")
