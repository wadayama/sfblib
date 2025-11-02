# IG_fig3_sfblib_vjp.py
# Fig.3-style comparison of dI/dalpha using sfblib v0.2.x with VJP helpers:
#   (1) Analytic gradient (closed-form)
#   (2) VJP with TRUE score
#   (3) VJP with DSM score (per-alpha training) + Stein scalar calibration
#
# Only one figure is saved: "IG_fig3_dIdalpha.pdf".
#
# Theory references:
# - Information-gradient & VJP identity: Eq.(10),(23),(25) in the info_grad paper.
# - DSM loss at fixed t: Eq.(27) in the MI paper (equivalent form here).
# - Linear vector channel analytic gradient: Eq.(83)-(84).

import math
import numpy as np
import torch
import matplotlib.pyplot as plt

import sfblib as sfb

# ------------------------------
# Problem setup (matches Fig.3)
# ------------------------------
seed = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sfb.set_seed(seed, deterministic=True)

n = m = 8            # dims
sigma_x2 = 1.0       # Var(X) = sigma_x2 * I
t = 0.5              # noise variance
alphas = np.linspace(0.0, 3.0, 31)  # alpha grid [0,3]

def sampler_x(batch, dev):
    return torch.randn(batch, n, device=dev) * math.sqrt(sigma_x2)

def orthogonal_matrix(n, dev):
    with torch.no_grad():
        q, r = torch.linalg.qr(torch.randn(n, n, device=dev))
        d = torch.sign(torch.diag(r))
        q = q @ torch.diag(d)
    return q

U = orthogonal_matrix(m, device)
V = orthogonal_matrix(n, device)
s_vals = torch.tensor(np.geomspace(3.0, 0.3, min(m, n)),
                      dtype=torch.float32, device=device)
A = U @ torch.diag(s_vals) @ V.T

# --------------------------------
# Front-ends: constant/parameter alpha
# --------------------------------
class FrontAlphaConst(torch.nn.Module):
    def __init__(self, A: torch.Tensor, alpha: float):
        super().__init__()
        self.register_buffer("A", A)
        self.alpha = float(alpha)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * (x @ self.A.T)

class FrontAlphaParam(torch.nn.Module):
    def __init__(self, A: torch.Tensor, alpha_init: float):
        super().__init__()
        self.register_buffer("A", A)
        self.alpha = torch.nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * (x @ self.A.T)

# --------------------------------
# (1) Analytic gradient (Eq.(84))
# --------------------------------
s2 = (s_vals**2).detach().cpu().numpy()
def grad_analytic(alpha: float) -> float:
    num = alpha * sigma_x2 * s2
    den = t + (alpha**2) * sigma_x2 * s2
    return float(np.sum(num / den))

# ---------------------------------------------------
# (2) VJP with TRUE score  s(y) = -Sigma_Y^{-1} y
# ---------------------------------------------------
def make_true_score(alpha: float):
    AX = A @ A.T
    SigmaY = (alpha**2) * sigma_x2 * AX + t * torch.eye(m, device=device)
    L = torch.linalg.cholesky(SigmaY)
    def s_true(y: torch.Tensor) -> torch.Tensor:
        # Solve SigmaY v = y  for each column of y^T (Cholesky), then negate
        v = torch.cholesky_solve(y.T, L, upper=False)   # (m,B)
        return -v.T
    return s_true

def grad_true_vjp(alpha: float, N: int = 100_000, batch: int = 8192) -> float:
    # Parameterized front-end with alpha as nn.Parameter to pick up d/dalpha via autograd
    frontend = FrontAlphaParam(A, alpha).to(device)
    out = sfb.estimate_info_grad(
        frontend=frontend,
        score_eval=make_true_score(alpha),
        sampler_x=sampler_x,
        t=t,
        N=N,
        batch_size=batch,
        device=device,
        params=(frontend.alpha,),      # gradient only for alpha
        stop_grad_score=True,          # stop-gradient on the score side (Eq.(25))
    )
    return float(out.get("alpha", list(out.values())[0]))

# ---------------------------------------------------
# (3) VJP with DSM score (per-alpha) + Stein scalar calibration
# ---------------------------------------------------
dsm = sfb.DSMConfig(steps=1000, batch_size=4096, lr=1e-3, hidden=256, layers=2, activation="silu", grad_clip=1.0)

def grad_dsm_vjp(alpha: float, N: int = 100_000, batch: int = 8192) -> float:
    # Train s_theta at this alpha (unconditional DSM at fixed t)
    frontend_train = FrontAlphaConst(A, alpha).to(device)
    score = sfb.train_dsm_uncond(sampler_x, frontend_train, t, dsm, device, y_dim=m)

    # Parameterized front-end for gradient extraction
    frontend = FrontAlphaParam(A, alpha).to(device)

    @torch.no_grad()
    def sample_y(B: int) -> torch.Tensor:
        x = sampler_x(B, device); z = torch.randn(B, m, device=device)
        return frontend(x) + math.sqrt(t) * z

    c = sfb.stein_calibrate_scalar(score, sample_y, m, B=8192)  # Gaussian Stein identity

    out = sfb.estimate_info_grad(
        frontend=frontend,
        score_eval=lambda y: c * score(y),
        sampler_x=sampler_x,
        t=t,
        N=N,
        batch_size=batch,
        device=device,
        params=(frontend.alpha,),
        stop_grad_score=True,
    )
    return float(out.get("alpha", list(out.values())[0]))

# ------------------------------
# Run and plot
# ------------------------------
def main():
    g_an, g_true, g_dsm = [], [], []
    print("Compute dI/dalpha on", device, "for alpha grid:", alphas)
    for a in alphas:
        print(f"[alpha={a:.2f}] analytic...", end="")
        g_an.append(grad_analytic(a))
        print(" true-score...", end="")
        g_true.append(grad_true_vjp(a, N=100_000, batch=8192))
        print(" DSM...", end="")
        g_dsm.append(grad_dsm_vjp(a, N=100_000, batch=8192))
        print(" done.")

    plt.figure()
    plt.plot(alphas, g_an, marker='o', label="Analytic dI/dalpha")
    plt.plot(alphas, g_true, marker='s', linestyle="--", label="VJP (TRUE score)")
    plt.plot(alphas, g_dsm, marker='^', linestyle="-.", label="VJP (DSM)")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\partial I / \partial \alpha$ [nats per unit]")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    out = "IG_fig3_dIdalpha_sfblib.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
