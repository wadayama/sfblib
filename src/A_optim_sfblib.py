
# A_optim_sfblib.py
# Projected gradient ascent for optimizing channel matrix A under Frobenius constraint.
#
# Problem setup:
#   Channel: Y = A X + Z_t
#   Input:   X ~ N(0, sigma_x2 * I_n)
#   Noise:   Z_t ~ N(0, t * I_m)
#   Constraint: ||A||_F <= P
#
# Method:
#   Alternating optimization via information-gradient VJP:
#   - Phase 1: Train unconditional DSM score at current A
#   - Phase 2: Estimate gradient dI/dA via VJP with Stein-calibrated score
#   - Phase 3: Projected gradient ascent step on A
#
# Output: PDF plot showing MI convergence vs theoretical optimum.
#
# Theory references:
# - Information-gradient VJP: Eq.(10),(23),(25) in the info_grad paper.
# - Projected gradient ascent: standard constrained optimization.

import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import sfblib as sfb

# ---------------- Config ----------------
SEED = 123
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sfb.set_seed(SEED, deterministic=True)

# Problem
n = m = 8             # n = m = 8
sigma_x2 = 1.0        # X ~ N(0, σ_x^2 I)
t = 0.5               # Z ~ N(0, t I)
P = 5.0               # Frobenius norm constraint ||A||_F <= P
outer_iters = 50      # number of alternating iterations

# DSM (unconditional; input is y only). Modest but stable defaults.
dsm = sfb.DSMConfig(steps=1000, batch_size=4096, lr=1e-3, hidden=256, layers=2, activation="silu", grad_clip=1.0)

# VJP/MC evaluation
N_mc = 50_000         # MC samples per gradient evaluation
batch_eval = 8192     # minibatch size for estimate_info_grad
step_size = 0.10      # gradient-ascent step size for A

SAVE_PDF = "A_optim_sfblib.pdf"

# ---------------- Utilities ----------------
def sampler_x(batch, dev):
    return torch.randn(batch, n, device=dev) * math.sqrt(sigma_x2)

# Front-ends
class LinearFrontConst(nn.Module):
    def __init__(self, A: torch.Tensor):
        super().__init__()
        self.register_buffer("A", A)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.A.T

class LinearFrontParam(nn.Module):
    def __init__(self, A_init: torch.Tensor):
        super().__init__()
        self.A = nn.Parameter(A_init.clone())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.A.T

# ---------------- Main loop ----------------
def main():
    # Init A with random Gaussian then scale to ||A||_F = P
    A = torch.randn(m, n, device=device)
    A = sfb.project_to_frobenius_ball(A, P)

    # Closed-form upper bound for isotropic optimum (Jensen/KKT)
    I_star = 0.5 * m * math.log(1.0 + (sigma_x2 / t) * (P**2 / m))

    I_hist, iters = [], []
    for k in range(outer_iters):
        # ---- Phase 1: train unconditional DSM score at current A ----
        front_train = LinearFrontConst(A.detach()).to(device)
        score = sfb.train_dsm_uncond(sampler_x, front_train, t, dsm, device, y_dim=m)

        # ---- Phase 2: estimate ∇_A I via VJP and update A ----
        front_eval = LinearFrontParam(A.detach()).to(device)

        @torch.no_grad()
        def sample_y(B: int) -> torch.Tensor:
            x = sampler_x(B, device); z = torch.randn(B, m, device=device)
            return front_eval(x) + math.sqrt(t) * z

        c = sfb.stein_calibrate_scalar(score, sample_y, m, B=8192)  # scalar calibration

        out = sfb.estimate_info_grad(
            frontend=front_eval,
            score_eval=lambda y: c * score(y),
            sampler_x=sampler_x,
            t=t,
            N=N_mc,
            batch_size=batch_eval,
            device=device,
            params=(front_eval.A,),   # take grad only w.r.t. A
            stop_grad_score=True,
        )
        # Gradient ascent step on A, then project to Frobenius ball
        gA = torch.tensor(out["A"], device=device, dtype=torch.float32)
        with torch.no_grad():
            front_eval.A += step_size * gA
            front_eval.A[:] = sfb.project_to_frobenius_ball(front_eval.A, P)
            A = front_eval.A.detach().clone()

        # Log MI
        I_hist.append(sfb.mi_linear_gaussian(A, sigma_x2, t))
        iters.append(k)

    # ---- Plot ----
    plt.figure()
    plt.plot(iters, I_hist, marker="o", label="Projected ascent (VJP + DSM, unconditional)")
    plt.axhline(I_star, linestyle="--", label="Theoretical optimum (isotropic case)")
    plt.xlabel("Iteration")
    plt.ylabel("Mutual information  I(A)  [nats]")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(SAVE_PDF)
    print(f"Saved: {SAVE_PDF}")

if __name__ == "__main__":
    main()
