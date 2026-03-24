
# MI_fm_identity.py
#
# Verification: Flow Matching MI Estimation for Identity Channel
# ==============================================================
# Validates the FM-based MI estimation by comparing against the closed-form
# solution for a Gaussian identity channel: I(X;Y_t) = 0.5 * n * log(1 + P/t).
#
# Two FM modes are tested:
#   1. FM noise-conditional (single model)
#   2. DSM per-t (baseline, for reference)
#
# Output: MI_fm_identity.pdf — FM estimate vs analytical MI curve
#
# Method: CFM velocity training → FlowMatchingScoreAdapter → Fisher MC → log-domain integration

import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sfblib as sfb

# -------------------------
# Settings
# -------------------------
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n, P = 4, 1.0
T_lower = P / 200.0
grid = sfb.LogGridConfig(t_min=T_lower, t_max=50.0 * P, m_points=12, T_lower=T_lower)

fm = sfb.FMConfig(steps=3000, batch_size=8192, lr=1e-3, grad_clip=1.0,
                  hidden=128, layers=3, activation="silu", t_embed_dim=32)
fisher = sfb.FisherConfig(mc_samples=100_000)
tail = sfb.TailConfig(use_tail=True, cov_trace_est_samples=50_000)

# For DSM baseline comparison
dsm = sfb.DSMConfig(steps=300, batch_size=8192, lr=1e-3, grad_clip=1.0, activation="silu")


def sampler_x(batch, dev):
    return torch.randn(batch, n, device=dev) * math.sqrt(P)


class Front(torch.nn.Module):
    def forward(self, x):
        return x


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
    sfb.set_seed(seed, deterministic=True)
    frontend = Front().to(device)
    t_vals = sfb.make_log_grid(grid)

    # --- Analytical solution ---
    I_true = 0.5 * n * np.log1p(P / t_vals)
    J_true = np.array([n / (P + t) for t in t_vals])

    # --- FM noise-conditional ---
    print("Training FM noise-conditional velocity model...")
    result_fm = sfb.estimate_mi_forward(
        sampler_x=sampler_x, frontend=frontend, t_grid=grid,
        dsm=dsm, fisher=fisher, tail=tail, device=device,
        conditional="fm_noise_cond", fm=fm,
    )
    J_hat_fm = result_fm["J_hat"]

    # Cumulative MI from FM
    tr_cov_x = sfb.estimate_trace_cov_x(sampler_x, device, samples=50_000)
    tail_val = 0.5 * (tr_cov_x / float(t_vals[-1]))
    I_hat_fm = cumulative_mi_log_trapz(t_vals, J_hat_fm, n, tail_val)

    # --- Fisher verification ---
    print("\nFisher information comparison (FM vs analytical):")
    print(f"{'t':>10s}  {'J_true':>10s}  {'J_hat_fm':>10s}  {'rel_err':>10s}")
    fisher_ok = True
    for i, t in enumerate(t_vals):
        rel_err = abs(J_hat_fm[i] - J_true[i]) / J_true[i]
        status = "OK" if rel_err < 0.05 else "WARN"
        print(f"{t:10.4f}  {J_true[i]:10.4f}  {J_hat_fm[i]:10.4f}  {rel_err:10.4f}  {status}")
        if rel_err >= 0.05:
            fisher_ok = False

    # --- MI verification ---
    mi_rel_err = abs(I_hat_fm[0] - I_true[0]) / I_true[0]
    print(f"\nMI at t_min={t_vals[0]:.4f}:")
    print(f"  I_true  = {I_true[0]:.6f} nats")
    print(f"  I_hat_fm = {I_hat_fm[0]:.6f} nats")
    print(f"  rel_err = {mi_rel_err:.6f}")
    mi_ok = mi_rel_err < 0.03

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(t_vals, I_true / n, 'k-o', label="Analytical: 0.5·log(1+P/t)", markersize=5)
    plt.plot(t_vals, I_hat_fm / n, 'r--^', label="FM noise-cond (per dim)", markersize=5)
    plt.xscale("log")
    plt.xlabel("Noise variance t")
    plt.ylabel("I(X; Y_t) / n  [nats]")
    plt.title("FM MI Estimation — Identity Channel Verification")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    out_path = "MI_fm_identity.pdf"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved figure to {out_path}")

    # --- Summary ---
    print("\n" + "=" * 50)
    if fisher_ok and mi_ok:
        print("PASS: FM MI estimation matches analytical solution.")
    else:
        if not fisher_ok:
            print("WARN: Some Fisher values exceeded 5% relative error.")
        if not mi_ok:
            print(f"WARN: MI relative error ({mi_rel_err:.4f}) exceeded 3% threshold.")
    print("=" * 50)


if __name__ == "__main__":
    main()
