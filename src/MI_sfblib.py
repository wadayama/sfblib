
# MI_sfblib.py
# Minimal, readable sample using sfblib v0.2.0.
# Output: a single PDF figure of mutual information vs. noise variance t.
#
# Method: DSM (fixed t) -> Fisher MC -> log-domain trapezoid integration with tail.
# (Fisher integral and log-domain trapezoid + tail follow the paper's Eq. (24), (43)–(45).)

import math
import numpy as np
import torch
import matplotlib.pyplot as plt

import sfblib

# -------------------------
# User settings (kept short)
# -------------------------
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n, P = 4, 1.0                                  # dim and input power
T_lower = P / 200.0                            # lower limit of the Fisher integral
grid = sfblib.LogGridConfig(t_min=T_lower, t_max=50.0 * P, m_points=12, T_lower=T_lower)

dsm = sfblib.DSMConfig(steps=300, batch_size=8192, lr=1e-3, grad_clip=1.0, activation="silu")
fisher = sfblib.FisherConfig(mc_samples=100_000)
tail = sfblib.TailConfig(use_tail=True, cov_trace_est_samples=50_000)

# X ~ N(0, P I_n)
def sampler_x(batch, dev):
    return torch.randn(batch, n, device=dev) * math.sqrt(P)

# Identity front-end: f(x) = x
class Front(torch.nn.Module):
    def forward(self, x): return x

# Log-domain cumulative integral helper:
# I_hat(t_k) = ∫_{t_k}^{t_max} 0.5*(n/t - J(t)) dt + tail
def cumulative_mi_log_trapz(t_vals: np.ndarray, J_vals: np.ndarray, n_dim: int, tail_val: float) -> np.ndarray:
    t = np.asarray(t_vals, dtype=np.float64)
    J = np.asarray(J_vals, dtype=np.float64)
    u = np.log(t)
    du = u[1] - u[0]
    g = 0.5 * (n_dim / t - J)              # integrand in t-domain
    integrand = g * t                      # g(e^u) * e^u
    area = 0.0
    I = np.empty_like(t)
    I[-1] = tail_val
    # integrate backward (from largest t down to each t_k)
    for k in range(len(t) - 2, -1, -1):
        area += 0.5 * (integrand[k+1] + integrand[k]) * du
        I[k] = area + tail_val
    return I

def main():
    sfblib.set_seed(seed, deterministic=True)
    frontend = Front().to(device)

    # (1) t-grid
    t_vals = sfblib.make_log_grid(grid)                      # numpy array, geometric spacing

    # (2) per-t DSM -> Fisher
    J_list = []
    for t in t_vals:
        score = sfblib.train_dsm_uncond(
            sampler_x=sampler_x, frontend=frontend, t=float(t),
            dsm=dsm, device=device, y_dim=n
        )
        Jt = sfblib.estimate_fisher_from_score(
            score=score, sampler_x=sampler_x, frontend=frontend,
            t=float(t), fisher=fisher, device=device
        )
        J_list.append(Jt)
    J_hat = np.asarray(J_list, dtype=np.float64)

    # (3) tail and cumulative MI
    tr_cov_x = sfblib.estimate_trace_cov_x(sampler_x, device, samples=tail.cov_trace_est_samples)
    tail_val = 0.5 * (tr_cov_x / float(t_vals[-1]))   # Eq. (45)
    I_hat = cumulative_mi_log_trapz(t_vals, J_hat, n, tail_val)

    # (4) reference (Gaussian input, identity front: closed form)
    I_true = 0.5 * n * np.log1p(P / t_vals)

    # (5) Plot ONLY the MI curve figure
    plt.figure()
    plt.plot(t_vals, I_true / n, marker='o', label="I_true/n = 0.5 log(1+P/t)")
    plt.plot(t_vals, I_hat / n, marker='s', linestyle='--', label="I_hat/n (DSM + tail)")
    plt.xscale("log")
    plt.xlabel("Noise variance t")
    plt.ylabel("I(X;Y_t)/n")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    out_path = f"MI_curve_P{str(P).replace('.', 'p')}.pdf"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved MI curve to {out_path}")

if __name__ == "__main__":
    main()
