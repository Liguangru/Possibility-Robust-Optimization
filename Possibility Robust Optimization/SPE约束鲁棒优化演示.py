# -*- coding: utf-8 -*-
"""
SPE（稀疏概率评估, de Jong 2026 PMB 71 095016）+ CVaR 约束鲁棒优化 方法学演示
框架：min OAR 的 upper-CVaR，s.t. Target 双侧 CVaR 约束（TXT 方案）
加速：SPE —— 预计算误差网格剂量；每分次误差映射到最近网格点；课程剂量对 spot 权重仍为线性
求解：Rockafellar-Uryasev 线性化 => 疗程层 + 体素层 CVaR 全部展开为辅助变量，整体是一个 LP（HiGHS 求解）
     Target 约束带 elastic 松弛（大惩罚），保证永远可解
注意：2D 方模体 + 玩具 Bragg 曲线，仅演示方法学，非临床剂量。
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

# ============ 1. 模体与结构 ============
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

# ============ 2. 玩具笔形束模型 ============
ISO = np.array([14.5, 13.5])
ANGLES = np.deg2rad([90.0, 210.0, 330.0])
SIG_LAT = 3.0                                            # 束斑横向 sigma (mm)
R_LAYERS = np.array([15.5, 18.0, 20.5, 23.0, 25.5, 28.0, 30.5])  # 7 个能量层
O_LAT = np.arange(-12.0, 12.1, 3.0)                      # 9 个横向位置
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

# ============ 3. SPE 误差网格 ============
s_sys, s_rand, s_rng = 0.92, 1.0, 1.5          # mm, mm, %
s_comb = np.hypot(s_sys, s_rand)
Emax = 3 * s_comb
g1d = np.linspace(-Emax, Emax, 7)              # order 7
gx, gy = np.meshgrid(g1d, g1d)
keep = (gx**2 + gy**2) <= Emax**2 + 1e-9
setup_grid = np.column_stack([gx[keep], gy[keep]])       # 29 点
range_grid = np.arange(-4.5, 4.51, 1.5)                  # 7 点
G = [(sx, sy, rr) for (sx, sy) in setup_grid for rr in range_grid]
n_grid = len(G)                                          # 203

t0 = time.time()
Dg = np.stack([struct_DI((sx, sy), rr) for (sx, sy, rr) in G])
t_pre = time.time() - t0
nom = next(i for i, (sx, sy, rr) in enumerate(G)
           if abs(sx) < 1e-12 and abs(sy) < 1e-12 and abs(rr) < 1e-12)
Dg_flat = Dg.reshape(n_grid, -1)

# ============ 4. 疗程场景采样 + 最近邻映射 ============
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
Tc = ((C / F) @ Dg_flat).reshape(K, V, n_spots)          # 每疗程结构 DI（对 x 线性）

# ============ 5. RU 线性化 LP（elastic 约束） ============
D_RX, DELTA, ALPHA = 60.0, 0.05, 0.10
LOW_FRAC, HOT_FRAC = 0.20, 0.05
ctv_sub = ctv_pos[np.sort(rng.choice(n_ctv, size=60, replace=False))]  # 体素子采样
nc = len(ctv_sub)
LAM = 200.0                                              # elastic 惩罚

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
         np.r_[-1., np.full(Kc, inv_aK), -1.]); adv(1, -D_RX*(1-delta))   # 低尾约束 + 松弛 e1
    for k in range(Kc):
        push(np.full(nc+3, r), np.r_[i_w+k, i_z+k, i_eL, i_u+k*nc+np.arange(nc)],
             np.r_[-1., -1., 1., np.full(nc, 1./nl)]); adv(1, 0.)
    push(np.full(Kc+2, r), np.r_[i_eH, i_q+np.arange(Kc), i_e2],
         np.r_[1., np.full(Kc, inv_aK), -1.]); adv(1, D_RX*(1+delta))     # 热尾约束 + 松弛 e2
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
res_d, _, t_det, _ = solve_lp(Dg[nom:nom+1], DELTA)      # 确定性对照（K=1，正确做法）
x_det = res_d.x[:n_spots]
print(f"鲁棒 LP:  {t_lp:5.1f} s  状态={res_r.message.strip()[:40]}")
print(f"  elastic 违反: 低尾 e1={res_r.x[i_e1]:.3f} Gy, 热尾 e2={res_r.x[i_e2]:.3f} Gy")
print(f"确定性 LP: {t_det:5.1f} s")

# ============ 6. 独立验证：SPE vs 精确逐分次计算 ============
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

print(f"\n=== 运行时间 ===")
print(f"SPE 网格预计算 ({n_grid} 个剂量):  {t_pre:5.1f} s")
print(f"SPE 评估 {Ke} 疗程:               {t_spe:5.2f} s")
print(f"精确评估 {Ke} 疗程 ({Ke*F} 分次):  {t_ex:5.1f} s   加速比 ~{t_ex/t_spe:.0f}x")
print(f"\n=== SPE 精度 (vs 精确, MPE) ===")
print(f"CTV 低尾 MPE: {mpe_cl:.3f} Gy | OAR 平均 MPE: {mpe_om:.3f} Gy")
print(f"\n=== 计划对比（独立精确评估, {Ke} 疗程）===")
print(f"{'指标':26s} {'确定性计划':>10s} {'SPE鲁棒计划':>12s}")
for lab, a, b in [("CTV低尾 CVaR_lower (>=57)", cvarl(cl_det), cvarl(cl_ex)),
                  ("CTV低尾 P10", np.percentile(cl_det, 10), np.percentile(cl_ex, 10)),
                  ("CTV低尾 最差疗程", cl_det.min(), cl_ex.min()),
                  ("CTV热尾 CVaR_upper (<=63)", cvaru(ch_det), cvaru(ch_ex)),
                  ("OAR平均 CVaR_upper (目标)", cvaru(om_det), cvaru(om_ex)),
                  ("OAR平均 P90", np.percentile(om_det, 90), np.percentile(om_ex, 90))]:
    print(f"{lab:26s} {a:10.2f} {b:12.2f}")

# ============ 7. 图 ============
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
ax.plot(*ISO, "k+"); ax.set_title("几何（红=CTV 蓝=OAR 黑点=SPE网格）"); ax.set_aspect("equal")
zz = np.linspace(0, 34, 400)
axs[1].plot(zz, bragg(zz, 24.0)); axs[1].set_title("玩具 Bragg 曲线 (R=24 mm)")
axs[1].set_xlabel("深度 (mm)")
im = axs[2].imshow(full_dose(x_rob), cmap="jet", origin="lower")
axs[2].contour(ctv, colors="white", linewidths=1.2)
axs[2].contour(oar, colors="cyan", linewidths=1.2)
axs[2].set_title("SPE 鲁棒计划名义剂量"); plt.colorbar(im, ax=axs[2], label="Gy")
fig.tight_layout(); fig.savefig(f"{OUT}/图1_几何与剂量模型.png", dpi=150); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
axs[0].plot(np.percentile(cl_ex, pct), pct, "k-", lw=2, label="精确（3500 分次）")
axs[0].plot(np.percentile(cl_spe, pct), pct, "r--", lw=2, label=f"SPE（{n_grid} 网格剂量）")
axs[0].axvline(D_RX * 0.95, color="gray", ls=":", label="约束下限 57 Gy")
axs[0].set_xlabel("CTV 低尾剂量 (Gy)"); axs[0].set_ylabel("百分位")
axs[0].set_title(f"CTV 覆盖 | MPE={mpe_cl:.2f} Gy"); axs[0].legend()
axs[1].plot(np.percentile(om_ex, pct), pct, "k-", lw=2, label="精确")
axs[1].plot(np.percentile(om_spe, pct), pct, "r--", lw=2, label="SPE")
axs[1].set_xlabel("OAR 平均剂量 (Gy)"); axs[1].set_ylabel("百分位")
axs[1].set_title(f"OAR | MPE={mpe_om:.2f} Gy"); axs[1].legend()
fig.tight_layout(); fig.savefig(f"{OUT}/图2_SPE精度验证.png", dpi=150); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
for v, lab, c_ in [(cl_det, "确定性计划", "tab:blue"), (cl_ex, "SPE 鲁棒计划", "tab:red")]:
    axs[0].plot(np.sort(v), np.linspace(0, 1, Ke), lw=2, color=c_, label=lab)
axs[0].axvline(D_RX * 0.95, color="gray", ls=":")
axs[0].set_xlabel("CTV 低尾剂量 (Gy)"); axs[0].set_ylabel("累积概率")
axs[0].set_title("靶区：最差尾部被抬高"); axs[0].legend()
for v, lab, c_ in [(om_det, "确定性计划", "tab:blue"), (om_ex, "SPE 鲁棒计划", "tab:red")]:
    axs[1].plot(np.sort(v), np.linspace(0, 1, Ke), lw=2, color=c_, label=lab)
axs[1].set_xlabel("OAR 平均剂量 (Gy)"); axs[1].set_ylabel("累积概率")
axs[1].set_title("OAR：鲁棒性的代价"); axs[1].legend()
fig.tight_layout(); fig.savefig(f"{OUT}/图3_鲁棒优化结果.png", dpi=150); plt.close(fig)
print("\n图已保存。")
