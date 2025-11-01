
# MI_tanh.py
#
# Experiment: MI Estimation for Nonlinear Tanh Channel
# ======================================================
# This script validates the Score-to-Fisher Bridge (SFB) methodology on a
# nonlinear channel and compares against KDE-LOO baseline estimator.
#
# Channel: Y_t = tanh(X * A^T) + sqrt(t) * Z
#   where A is an orthogonal matrix, X ~ N(0, P*I), Z ~ N(0, I)
#
# Methods compared:
# 1. DSM -> Fisher -> Integration (proposed SFB method)
# 2. KDE-LOO with Gaussian kernel (baseline from Eq. 50)
#
# Paper correspondence:
#   - Reproduces Figure 6 (tanh channel MI curves)
#   - Implements Eq. (24): Fisher integral
#   - Implements Eq. (43)-(45): Log-domain trapezoid + tail
#   - Implements Eq. (27): DSM for score learning
#   - Implements Eq. (50): KDE-LOO baseline estimator
#
# Output: MI_tanh_fig6.pdf - Plot comparing DSM vs KDE-LOO across noise levels  

import math
import numpy as np
import torch
import matplotlib.pyplot as plt

import sfblib as sfb

# -----------------------
# Basic configuration
# -----------------------
seed = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sfb.set_seed(seed, deterministic=True)

n, P = 4, 1.0
T_lower = P / 200.0
grid = sfb.LogGridConfig(t_min=T_lower, t_max=50.0 * P, m_points=12, T_lower=T_lower)

# DSM/Fisher settings
dsm = sfb.DSMConfig(steps=400, batch_size=8192, lr=1e-3, grad_clip=1.0, activation="silu")
fisher = sfb.FisherConfig(mc_samples=100_000)

# KDE settings
N_kde = 20_000            # as used in the paper for the baseline (full-sum, no kNN).  
kde_chunk = 4096          # to limit memory in the LOO log-sum-exp if needed

# -----------------------
# Source and frontend
# -----------------------

def sampler_x(batch, dev):
    return torch.randn(batch, n, device=dev) * math.sqrt(P)

def orthogonal_matrix(n, dev):
    with torch.no_grad():
        q, r = torch.linalg.qr(torch.randn(n, n, device=dev))
        # enforce a proper rotation (det=+1) by flipping sign if needed
        d = torch.sign(torch.diag(r))
        q = q @ torch.diag(d)
    return q

class TanhFront(torch.nn.Module):
    def __init__(self, A: torch.Tensor):
        super().__init__()
        self.register_buffer("A", A)  # (n,n)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x @ self.A.T)

A = orthogonal_matrix(n, device)
frontend = TanhFront(A).to(device)

# -----------------------
# Helper: cumulative MI from J(t) curve (log-domain trapezoid)
# I_hat(t_k) = ∫_{t_k}^{t_max} 0.5 * (n/t - J(t)) dt + tail (Eq.(43)-(45))  
# -----------------------
def cumulative_mi_log_trapz(t_vals: np.ndarray, J_vals: np.ndarray, n_dim: int, tail_val: float) -> np.ndarray:
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
        area += 0.5 * (integrand[k+1] + integrand[k]) * du
        I[k] = area + tail_val
    return I

# -----------------------
# Run: DSM->Fisher per t, then cumulative MI; plus KDE-LOO baseline
# -----------------------
def main():
    # t-grid
    t_vals = sfb.make_log_grid(grid)

    # DSM/Fisher route
    J_list = []
    for t in t_vals:
        score = sfb.train_dsm_uncond(sampler_x, frontend, float(t), dsm, device, y_dim=n)
        Jt = sfb.estimate_fisher_from_score(score, sampler_x, frontend, float(t), fisher, device)
        J_list.append(Jt)
    J_hat = np.asarray(J_list, dtype=np.float64)

    # tail term (Eq.(45)) using tr Cov(X)
    tr_cov_x = sfb.estimate_trace_cov_x(sampler_x, device, samples=50_000)
    tail_val = 0.5 * (tr_cov_x / float(t_vals[-1]))
    I_hat = cumulative_mi_log_trapz(t_vals, J_hat, n, tail_val)  # DSM curve

    # KDE-LOO baseline (Eq.(50)) at each t (independent estimates)
    I_kde = []
    for t in t_vals:
        Ik = sfb.estimate_mi_kde_loo(sampler_x, frontend, float(t), N=N_kde, device=device, chunk=kde_chunk)
        I_kde.append(Ik)
    I_kde = np.asarray(I_kde, dtype=np.float64)

    # Plot ONLY the MI curves (per dimension)
    plt.figure()
    plt.plot(t_vals, I_kde / n, marker='o', label="KDE-LOO baseline (per dim)")
    plt.plot(t_vals, I_hat / n, marker='^', linestyle='--', label="DSM (proposed, per dim)")
    plt.xscale("log")
    plt.xlabel("Noise variance t")
    plt.ylabel("I(X;Y_t)/n")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    out_path = "MI_tanh_fig6.pdf"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved MI curve to {out_path}")

if __name__ == "__main__":
    main()
