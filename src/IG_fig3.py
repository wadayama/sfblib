
# IG_fig3_v020.py
# Information-gradient sample for Fig.3 in the info_grad paper:
# Compare dI/dalpha for a linear vector Gaussian channel using three methods:
#   (1) Analytic gradient (closed-form)
#   (2) VJP with TRUE score  s_Y(y) = -Sigma_Y^{-1} y     (Monte Carlo)
#   (3) VJP with learned score via DSM (per-alpha training, unconditional)
#
# References:
# - Information-gradient formula & VJP identity: Eq.(10),(23),(25).  fileciteturn0file1
# - DSM loss (fixed t): Eq.(27) in MI paper (equiv. Eq.(61) style here).      fileciteturn0file0
# - Linear vector channel analytic gradient: Eq.(83)-(84).                    fileciteturn0file1
#
# Output: A single PDF figure "IG_fig3_dIdalpha.pdf".
#
# NOTE: This script uses sfblib v0.2.x (single-file). Place sfblib.py next to this file.
#       The DSM configuration below matches the paper's simple MLP (2 layers, width 256).

import math
import numpy as np
import torch
import matplotlib.pyplot as plt

import sfblib as sfb

# ------------------------------
# Basic problem (matches Fig.3)
# ------------------------------
seed = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sfb.set_seed(seed, deterministic=True)

n = m = 8            # dims
sigma_x2 = 1.0       # Var(X) = sigma_x2 * I
t = 0.5              # noise variance
alphas = np.linspace(0.0, 3.0, 31)  # alpha grid (0..3)

# mild ill-conditioning for A (largest/smallest ~= 12)
def orthogonal_matrix(n, dev):
    with torch.no_grad():
        q, r = torch.linalg.qr(torch.randn(n, n, device=dev))
        d = torch.sign(torch.diag(r))
        q = q @ torch.diag(d)
    return q

U = orthogonal_matrix(m, device)
V = orthogonal_matrix(n, device)
ratio = 12.0
s_max, s_min = 1.0, 1.0/ratio
s_vals = torch.logspace(math.log10(s_max), math.log10(s_min), steps=min(m, n), device=device)
A = U @ torch.diag(s_vals) @ V.T  # (m,n)

# --------------------------------
# Utilities: samplers and frontends
# --------------------------------
def sampler_x(batch, dev):
    return torch.randn(batch, n, device=dev) * math.sqrt(sigma_x2)

class FrontAlphaConst(torch.nn.Module):
    """f(x) = alpha A x (alpha is a Python float; no grad needed)."""
    def __init__(self, A: torch.Tensor, alpha: float):
        super().__init__()
        self.register_buffer("A", A)
        self.alpha = float(alpha)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * (x @ self.A.T)

class FrontAlphaParam(torch.nn.Module):
    """f(x) = alpha A x with alpha as nn.Parameter (to take d/dalpha via autograd)."""
    def __init__(self, A: torch.Tensor, alpha_init: float):
        super().__init__()
        self.register_buffer("A", A)
        self.alpha = torch.nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * (x @ self.A.T)

# --------------------------------
# (1) Analytic gradient (Eq.(84))
# --------------------------------
# dI/dalpha = Sigma_i alpha sigma_x^2 s_i^2 / ( t + alpha^2 sigma_x^2 s_i^2 )
s2 = (s_vals**2).detach().cpu().numpy()
def grad_analytic(alpha: float) -> float:
    num = alpha * sigma_x2 * s2
    den = t + (alpha**2) * sigma_x2 * s2
    return float(np.sum(num / den))

# ---------------------------------------------------
# (2) VJP with TRUE score s_Y(y) = - Sigma_Y^{-1} y (MC)
# ---------------------------------------------------
# dI/dalpha = - E[ < A X , s_Y(Y) > ],   Y = alpha A X + Z,  Sigma_Y = alpha^2 sigma_x^2 A A^T + t I
def grad_true_score(alpha: float, N: int = 100_000, chunk: int = 16_384) -> float:
    with torch.no_grad():
        AX = A @ A.T  # (m,m)
        SigmaY = (alpha**2) * sigma_x2 * AX + t * torch.eye(m, device=device)
        # Precompute a solver via Cholesky for stability
        L = torch.linalg.cholesky(SigmaY)  # SigmaY = L L^T

        ssum = 0.0
        done = 0
        while done < N:
            B = min(chunk, N - done)
            x = sampler_x(B, device)           # (B,n)
            z = torch.randn(B, m, device=device)
            ax = x @ A.T                        # (B,m) = A x
            y = alpha * ax + z                  # Y
            # s(Y) = - SigmaY^{-1} Y = - solve(L L^T, Y)
            # Solve two triangular systems: L u = y^T -> u; L^T v = u -> v  (v = Sigma^{-1} y^T)
            u = torch.cholesky_solve(y.T, L, upper=False)  # (m,B)
            s = -u.T                            # (B,m)
            ssum += float((ax * s).sum().item())
            done += B
        dot_mean = ssum / N
        return -dot_mean

# ---------------------------------------------------
# (3) VJP with learned score via DSM (per-alpha training)
# ---------------------------------------------------
# DSM config (close to the paper: 2 layers, width 256, steps ~1000)
dsm = sfb.DSMConfig(steps=1000, batch_size=4096, lr=1e-3, hidden=256, layers=2, activation="silu", grad_clip=1.0)

def stein_calibrate(score_model: torch.nn.Module, sampler_y, B: int = 8192) -> float:
    """Scalar c so that E[Y^T (c s(Y))] ~= -m (Gaussian Stein identity)."""
    with torch.no_grad():
        y = sampler_y(B)
        s = score_model(y)
        den = (y * s).sum(dim=1).mean().item()
        return float(-m / (den + 1e-12))

def grad_dsm(alpha: float, N_eval: int = 100_000, chunk: int = 8192) -> float:
    # Train sθ at this alpha (unconditional DSM at fixed t)
    frontend_train = FrontAlphaConst(A, alpha).to(device)
    score = sfb.train_dsm_uncond(sampler_x, frontend_train, t, dsm, device, y_dim=m)

    # Prepare alpha as a differentiable parameter for the VJP step
    frontend_eval = FrontAlphaParam(A, alpha).to(device)
    alpha_param = frontend_eval.alpha

    # y-sampler for Stein calibration
    @torch.no_grad()
    def sample_y(B: int) -> torch.Tensor:
        x = sampler_x(B, device)
        z = torch.randn(B, m, device=device)
        return frontend_eval(x) + z * math.sqrt(t)

    c = stein_calibrate(score, sample_y, B=8192)

    # Accumulate the gradient of Lvjp = E < falpha(x), stop(sθ(y)) >
    # dI/dalpha ~= - d/dalpha Lvjp
    if alpha_param.grad is not None:
        alpha_param.grad.zero_()

    done = 0
    while done < N_eval:
        B = min(chunk, N_eval - done)
        x = sampler_x(B, device)
        z = torch.randn(B, m, device=device)
        f = frontend_eval(x)                    # alpha A x
        y = f + z * math.sqrt(t)
        s = (c * score(y)).detach()            # stop-gradient
        L = (f * s).sum(dim=1).mean()          # inner product average
        w = B / float(N_eval)
        (w * L).backward()                     # accumulate into alpha.grad
        done += B

    grad_est = - float(alpha_param.grad.item())
    # clean residual graph refs
    alpha_param.grad = None
    return grad_est

# ------------------------------
# Run the three estimators
# ------------------------------
def main():
    g1, g2, g3 = [], [], []
    print("Computing dI/dalpha on", device, "for alpha in", alphas)
    for a in alphas:
        print(f"[alpha={a:.2f}] analytic...", end="")
        g1.append(grad_analytic(a))
        print(" true-score...", end="")
        g2.append(grad_true_score(a, N=100_000, chunk=16_384))
        print(" DSM...", end="")
        g3.append(grad_dsm(a, N_eval=100_000, chunk=8192))
        print(" done.")

    # Plot
    plt.figure()
    plt.plot(alphas, g1, label="Analytic dI/dalpha")
    plt.plot(alphas, g2, linestyle="--", label="VJP (TRUE score)")
    plt.plot(alphas, g3, linestyle="-.", label="VJP (DSM, unconditional)")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\partial I / \partial \alpha$  [nats per unit]")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    out = "IG_fig3_dIdalpha.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
