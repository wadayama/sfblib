
# MI_fm_tanh.py
#
# Verification: Flow Matching MI for Nonlinear Tanh Channel
# ============================================================
# No closed-form solution exists, so we compare three methods:
#   1. DSM per-t (existing SFB baseline)
#   2. FM noise-conditional (new flow matching approach)
#   3. KDE-LOO (independent reference estimator)
#
# Additionally, we directly compare the score functions from DSM and FM
# at each grid point using relative MSE.
#
# Output: MI_fm_tanh.pdf — Three-method MI curve comparison
#
# Channel: Y_t = tanh(X * A^T) + sqrt(t) * Z
#   where A is orthogonal, X ~ N(0, P*I), Z ~ N(0, I)

import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sfblib as sfb

# -------------------------
# Settings
# -------------------------
seed = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sfb.set_seed(seed, deterministic=True)

n, P = 4, 1.0
T_lower = P / 200.0
grid = sfb.LogGridConfig(t_min=T_lower, t_max=50.0 * P, m_points=12, T_lower=T_lower)

dsm = sfb.DSMConfig(steps=400, batch_size=8192, lr=1e-3, grad_clip=1.0, activation="silu")
fm = sfb.FMConfig(steps=3000, batch_size=8192, lr=1e-3, grad_clip=1.0,
                  hidden=128, layers=3, activation="silu", t_embed_dim=32)
fisher = sfb.FisherConfig(mc_samples=100_000)
tail = sfb.TailConfig(use_tail=True, cov_trace_est_samples=50_000)

N_kde = 20_000
kde_chunk = 4096

# -------------------------
# Source and frontend
# -------------------------

def sampler_x(batch, dev):
    return torch.randn(batch, n, device=dev) * math.sqrt(P)


def orthogonal_matrix(n_dim, dev):
    with torch.no_grad():
        q, r = torch.linalg.qr(torch.randn(n_dim, n_dim, device=dev))
        d = torch.sign(torch.diag(r))
        q = q @ torch.diag(d)
    return q


class TanhFront(nn.Module):
    def __init__(self, A: torch.Tensor):
        super().__init__()
        self.register_buffer("A", A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x @ self.A.T)


# Cumulative MI helper
def cumulative_mi_log_trapz(t_vals, J_vals, n_dim, tail_val):
    t = np.asarray(t_vals, dtype=np.float64)
    J = np.asarray(J_vals, dtype=np.float64)
    u = np.log(t)
    du = u[1] - u[0]
    g = 0.5 * (n_dim / t - J)
    integrand = g * t
    area = 0.0
    I = np.empty_like(t)
    I[-1] = tail_val
    for k in range(len(t) - 2, -1, -1):
        area += 0.5 * (integrand[k + 1] + integrand[k]) * du
        I[k] = area + tail_val
    return I


def main():
    A = orthogonal_matrix(n, device)
    frontend = TanhFront(A).to(device)
    t_vals = sfb.make_log_grid(grid)

    # ----------------------------------------------------------
    # 1. DSM per-t (baseline)
    # ----------------------------------------------------------
    print("=== DSM per-t (baseline) ===")
    J_dsm = []
    dsm_score_models = []
    for i, t in enumerate(t_vals):
        score = sfb.train_dsm_uncond(sampler_x, frontend, float(t), dsm, device, y_dim=n)
        Jt = sfb.estimate_fisher_from_score(score, sampler_x, frontend, float(t), fisher, device)
        J_dsm.append(Jt)
        dsm_score_models.append(score)
        print(f"  t={t:.4f}  J_dsm={Jt:.4f}")
    J_dsm = np.array(J_dsm, dtype=np.float64)

    tr_cov_x = sfb.estimate_trace_cov_x(sampler_x, device, samples=50_000)
    tail_val = 0.5 * (tr_cov_x / float(t_vals[-1]))
    I_dsm = cumulative_mi_log_trapz(t_vals, J_dsm, n, tail_val)

    # ----------------------------------------------------------
    # 2. FM noise-conditional
    # ----------------------------------------------------------
    print("\n=== FM noise-conditional ===")
    vel_net = sfb.train_fm_noise_cond(sampler_x, frontend, float(grid.t_min), float(grid.t_max),
                                       fm, device, y_dim=n)
    J_fm = []
    fm_adapters = []
    for i, t in enumerate(t_vals):
        tau = sfb.t_to_tau(float(t))
        adapter = sfb.FlowMatchingScoreAdapter(vel_net, tau)
        Jt = sfb.estimate_fisher_from_score(adapter, sampler_x, frontend, float(t), fisher, device)
        J_fm.append(Jt)
        fm_adapters.append(adapter)
        print(f"  t={t:.4f}  J_fm={Jt:.4f}")
    J_fm = np.array(J_fm, dtype=np.float64)
    I_fm = cumulative_mi_log_trapz(t_vals, J_fm, n, tail_val)

    # ----------------------------------------------------------
    # 3. KDE-LOO (independent reference)
    # ----------------------------------------------------------
    print("\n=== KDE-LOO (reference) ===")
    I_kde = []
    for t in t_vals:
        Ik = sfb.estimate_mi_kde_loo(sampler_x, frontend, float(t), N=N_kde,
                                      device=device, chunk=kde_chunk)
        I_kde.append(Ik)
        print(f"  t={t:.4f}  I_kde={Ik:.4f}")
    I_kde = np.array(I_kde, dtype=np.float64)

    # ----------------------------------------------------------
    # 4. Score comparison: DSM vs FM
    # ----------------------------------------------------------
    print("\n=== Score comparison (relative MSE: FM vs DSM) ===")
    print(f"{'t':>10s}  {'rel_MSE':>10s}  {'status':>6s}")
    n_score_samples = 10_000
    score_ok = True
    for i, t in enumerate(t_vals):
        with torch.no_grad():
            x = sampler_x(n_score_samples, device)
            w = frontend(x)
            y = w + torch.randn_like(w) * math.sqrt(float(t))
            s_dsm = dsm_score_models[i](y)
            s_fm = fm_adapters[i](y)
            mse = torch.mean((s_fm - s_dsm) ** 2).item()
            ref = torch.mean(s_dsm ** 2).item()
            rel_mse = mse / max(ref, 1e-30)
        status = "OK" if rel_mse < 0.10 else "WARN"
        print(f"{float(t):10.4f}  {rel_mse:10.4f}  {status}")
        if rel_mse >= 0.10:
            score_ok = False

    # ----------------------------------------------------------
    # 5. Plot
    # ----------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(t_vals, I_kde / n, 'bo-', label="KDE-LOO (per dim)", markersize=5)
    plt.plot(t_vals, I_dsm / n, 'gs--', label="DSM per-t (per dim)", markersize=5)
    plt.plot(t_vals, I_fm / n, 'r^--', label="FM noise-cond (per dim)", markersize=5)
    plt.xscale("log")
    plt.xlabel("Noise variance t")
    plt.ylabel("I(X; Y_t) / n  [nats]")
    plt.title("MI Estimation — Tanh Channel: DSM vs FM vs KDE-LOO")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    out_path = "MI_fm_tanh.pdf"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved figure to {out_path}")

    # ----------------------------------------------------------
    # 6. Summary
    # ----------------------------------------------------------
    print("\n" + "=" * 50)
    print(f"MI at t_min={t_vals[0]:.4f}:")
    print(f"  I_dsm  = {I_dsm[0]:.4f} nats")
    print(f"  I_fm   = {I_fm[0]:.4f} nats")
    print(f"  I_kde  = {I_kde[0]:.4f} nats")
    mi_fm_dsm_rel = abs(I_fm[0] - I_dsm[0]) / max(abs(I_dsm[0]), 1e-30)
    print(f"  |I_fm - I_dsm| / |I_dsm| = {mi_fm_dsm_rel:.4f}")

    if score_ok:
        print("PASS: Score relative MSE < 0.10 at all grid points.")
    else:
        print("WARN: Some score comparisons exceeded 10% relative MSE.")
    print("=" * 50)


if __name__ == "__main__":
    main()
